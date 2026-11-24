"""Frozen-encoder math information probes; never claims VAE reconstruction accuracy.

Synthetic arithmetic groups are separated by unordered operand pair. Validation
selects ridge strength; final evaluation uses unseen pairs and a new template.
Raw-backbone and shuffled-label controls use the same examples/protocol.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

OPERATIONS=('add','subtract','multiply','divide')


def expression(a,b,operation,template):
    symbols={'add':'+','subtract':'-','multiply':'*','divide':'/'}
    if template==0:return f'Compute {a} {symbols[operation]} {b}.'
    if template==1:
        return {'add':f'Add {a} and {b}.','subtract':f'Subtract {b} from {a}.',
                'multiply':f'Multiply {a} by {b}.','divide':f'Divide {a} by {b}.'}[operation]
    if template==2:
        return {'add':f'The sum of {a} and {b} is required.',
                'subtract':f'The difference of {a} and {b} is required.',
                'multiply':f'The product of {a} and {b} is required.',
                'divide':f'The quotient of {a} and {b} is required.'}[operation]
    if template==3:
        return {'add':f'Starting with {a}, add {b}.',
                'subtract':f'Starting with {a}, take away {b}.',
                'multiply':f'Find the product when {a} is multiplied by {b}.',
                'divide':f'How much is {a} divided by {b}?'}[operation]
    raise ValueError(template)


def controlled_data(seed=104729):
    rng=random.Random(seed)
    pairs={(a,b) for a in range(1,21) for b in range(a+1,21)}
    # Three disjoint cycles ensure every value appears in every split.
    cycles=[];used=set()
    for _ in range(3):
        for attempt in range(10000):
            order=list(range(1,21));rng.shuffle(order)
            cycle={tuple(sorted((order[i],order[(i+1)%20]))) for i in range(20)}
            if len(cycle)==20 and not cycle&used:
                cycles.append(cycle);used|=cycle;break
        else:raise RuntimeError('Could not build disjoint, value-covering evaluation groups')
    remainder=sorted(pairs-used);rng.shuffle(remainder)
    selections={'train':cycles[0]|set(remainder[:60]),'validation':cycles[1],'heldout':cycles[2]}
    rows=[]
    for split,selected in selections.items():
        for first,second in sorted(selected):
            for a,b in [(first,second),(second,first)]:
                for operation in OPERATIONS:
                    template=rng.randrange(3) if split!='heldout' else 3
                    rows.append({'text':expression(a,b,operation,template),'a':a,'b':b,
                                 'operation':operation,'template':template,'split':split,
                                 'group':[first,second]})
    groups={split:{tuple(r['group']) for r in rows if r['split']==split}
            for split in ['train','validation','heldout']}
    assert not (groups['train']&groups['validation'] or groups['train']&groups['heldout'] or
                groups['validation']&groups['heldout'])
    return rows


def alternatives(row):
    a,b,op=row['a'],row['b'],row['operation']
    other=OPERATIONS[(OPERATIONS.index(op)+1)%len(OPERATIONS)]
    return [expression(a,b,op,0),expression(a%20+1,b,op,0),
            expression(a,b%20+1,op,0),expression(a,b,other,0)]


def classification_metrics(scores,labels):
    import torch
    cuts=[(0,20),(20,40),(40,44)]
    prediction=torch.stack([scores[:,a:b].argmax(-1) for a,b in cuts],dim=1)
    hits=prediction.eq(labels)
    values={'examples':len(labels),'first_operand_accuracy':float(hits[:,0].float().mean()),
            'second_operand_accuracy':float(hits[:,1].float().mean()),
            'operation_accuracy':float(hits[:,2].float().mean()),
            'joint_exact_accuracy':float(hits.all(-1).float().mean()),
            'joint_exact_count':int(hits.all(-1).sum())}
    return values,prediction.cpu(),hits.all(-1).cpu()


def ridge_probe(features,indices,labels,device,shuffle=False):
    import torch
    import torch.nn.functional as F
    train=features[indices['train']].to(device=device,dtype=torch.float32)
    validation=features[indices['validation']].to(device=device,dtype=torch.float32)
    heldout=features[indices['heldout']].to(device=device,dtype=torch.float32)
    mean=train.mean(0);scale=train.std(0,unbiased=False).clamp_min(1e-3)
    train=(train-mean)/scale;validation=(validation-mean)/scale;heldout=(heldout-mean)/scale
    target=labels[indices['train']].to(device)
    if shuffle:
        generator=torch.Generator().manual_seed(8675309)
        target=target[torch.randperm(len(target),generator=generator).to(device)]
    y=torch.cat([F.one_hot(target[:,0],20),F.one_hot(target[:,1],20),F.one_hot(target[:,2],4)],dim=1).float()
    prior=y.mean(0);y=y-prior
    dimension=train.shape[1]
    kernel=train@train.T/dimension
    val_kernel=validation@train.T/dimension
    test_kernel=heldout@train.T/dimension
    identity=torch.eye(len(train),device=device)
    best=None;selection=[]
    for regularization in [.0001,.001,.01,.1,1.,10.]:
        dual=torch.linalg.solve(kernel+regularization*identity,y)
        metrics,_,_=classification_metrics(val_kernel@dual+prior,labels[indices['validation']].to(device))
        key=(metrics['joint_exact_accuracy'],sum(metrics[k] for k in
              ['first_operand_accuracy','second_operand_accuracy','operation_accuracy']))
        selection.append({'ridge':regularization,**metrics})
        if best is None or key>best[0]:best=(key,regularization,dual)
    metrics,predictions,hits=classification_metrics(test_kernel@best[2]+prior,labels[indices['heldout']].to(device))
    model={'training_mean':mean.cpu(),'training_scale':scale.cpu(),'training_features':train.cpu(),
           'dual':best[2].cpu(),'prior':prior.cpu(),'ridge':best[1]}
    return {'heldout':metrics,'selected_ridge':best[1],'validation_grid':selection,
            'feature_dimension':dimension,'shuffled_training_labels':shuffle},predictions,hits,model


def equivalent_expression_metrics(prediction,labels):
    """Accept operand swaps only for addition and multiplication."""
    import torch
    exact=prediction.eq(labels).all(-1)
    commutative=(labels[:,2]==0)|(labels[:,2]==2)
    swapped=(prediction[:,0]==labels[:,1])&(prediction[:,1]==labels[:,0])&(prediction[:,2]==labels[:,2])
    correct=exact|(commutative&swapped)
    return {'mathematically_equivalent_expression_count':int(correct.sum()),
            'mathematically_equivalent_expression_accuracy':float(correct.float().mean())}


def familiar_wording_diagnostic(output,device='cuda:1'):
    """Apply existing readouts to the same unseen pairs in familiar notation.

    No fit, ridge selection, or label changes occur. This diagnostic was added
    after seeing the primary new-wording result and is reported separately.
    """
    import torch
    output=Path(output)
    saved=torch.load(output/'features.pt',map_location='cpu',weights_only=True,mmap=True)
    features=saved['features']
    rows=[json.loads(line) for line in (output/'readout-predictions.jsonl').open()]
    alternatives=[json.loads(line) for line in (output/'retrieval.jsonl').open()]
    assert len(rows)==len(alternatives)
    assert all(a['feature_index']==b['anchor_index'] for a,b in zip(rows,alternatives))
    indices=torch.tensor([r['candidate_indices'][0] for r in alternatives])
    labels=torch.tensor([[r['a']-1,r['b']-1,OPERATIONS.index(r['operation'])] for r in rows])
    results={};predictions=[]
    for name in ['full_latents','pooled_latents','backbone_text','shuffled_labels']:
        source='full_latents' if name=='shuffled_labels' else name
        model=torch.load(output/f'{name}-readout.pt',map_location='cpu',weights_only=True,mmap=True)
        selected=features[source][indices].to(device)
        selected=(selected-model['training_mean'].to(device))/model['training_scale'].to(device)
        kernel=selected@model['training_features'].to(device).T/selected.shape[1]
        scores=kernel@model['dual'].to(device)+model['prior'].to(device)
        metrics,prediction,_=classification_metrics(scores,labels.to(device))
        metrics.update(equivalent_expression_metrics(prediction,labels))
        results[name]=metrics
        predictions.append((name,prediction.tolist()))
        del selected,kernel,scores,model
    with (output/'familiar-wording-predictions.jsonl').open('w') as stream:
        for i,(row,options) in enumerate(zip(rows,alternatives)):
            stream.write(json.dumps({'text':options['candidates'][0],'a':row['a'],'b':row['b'],
                 'operation':row['operation'],'predicted_classes':{name:values[i] for name,values in predictions}})+'\n')
    summary=json.loads((output/'summary.json').read_text())
    for name in summary['readouts']:
        prediction=torch.tensor([r['predicted_classes'][name] for r in rows])
        summary['readouts'][name]['heldout'].update(equivalent_expression_metrics(prediction,labels))
    summary['familiar_wording_readout_diagnostic']={
        'posthoc_diagnostic':True,'same_unseen_operand_pairs_as_primary':True,
        'encoder_or_readout_refitted':False,'same_validation_selected_regularization':True,
        'wording':'Compute a OP b.; template zero is present in readout training',
        'results':results}
    summary['posthoc_diagnostic_implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    temporary=output/'summary.json.tmp';temporary.write_text(json.dumps(summary,indent=2)+'\n');temporary.replace(output/'summary.json')
    print(json.dumps(summary['familiar_wording_readout_diagnostic'],indent=2),flush=True)
    return summary


def paired_significance(first,second):
    better=int((first&~second).sum());worse=int((~first&second).sum());n=better+worse
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1.
    return {'latent_only_correct':better,'baseline_only_correct':worse,
            'two_sided_exact_paired_binomial_p':p}


def run_evaluation(root,encoder_device='cuda:0',probe_device='cuda:1'):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from .released_encoder import VAEEncoder,positions_from_mask
    from .adaptive_batches import padded_batch
    root=Path(root)
    name='pretrained-vae-quality-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output=root/'outputs'/name;output.mkdir()
    index=root/'manifests/pretrained_vae_quality.json'
    def write(path,value):
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(path)
    start=time.monotonic();state={'run_directory':str(output),'stage':'preparing','encoder_training':False}
    def status(**values):
        state.update(values,elapsed_seconds=time.monotonic()-start,
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        write(index,state);write(output/'status.json',state);print(json.dumps(state),flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False
        rows=controlled_data()
        texts=[];lookup={}
        def register(text):
            if text not in lookup:lookup[text]=len(texts);texts.append(text)
            return lookup[text]
        for row in rows:row['feature_index']=register(row['text'])
        retrieval=[]
        for row in rows:
            if row['split']=='heldout':
                retrieval.append({'anchor':row['text'],'anchor_index':row['feature_index'],
                                  'candidates':alternatives(row),
                                  'candidate_indices':[register(text) for text in alternatives(row)],
                                  'correct_candidate':0,'group':row['group']})
        real=[]
        numeric=re.compile(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])')
        for dataset in ['gsm8k','dart-math-hard']:
            seen=set();count=0
            with (root/f'data/processed/math-sentences-v1/sentences/{dataset}.dev.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line);text=row['reasoning_block'];match=numeric.search(text)
                    if not match or row['problem_id'] in seen or len(text)>360:continue
                    value=float(match.group());replacement=str(int(value)+1) if value.is_integer() else f'{value+1:g}'
                    changed=text[:match.start()]+replacement+text[match.end():]
                    real.append({'dataset':dataset,'problem_id':row['problem_id'],'original':text,
                                 'changed':changed,'original_index':register(text),'changed_index':register(changed),
                                 'edit':'first numeric literal increased by one; mathematical validity not assumed'})
                    seen.add(row['problem_id']);count+=1
                    if count==32:break
        for filename,value in [('controlled.jsonl',rows),('retrieval.jsonl',retrieval),('real-number-edits.jsonl',real)]:
            with (output/filename).open('w') as stream:
                for row in value:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        protocol={'seed':104729,'encoder_frozen':True,'inference_precision':'FP32 from published BF16 weights',
                  'controlled_examples':len(rows),'encoding_texts':len(texts),'real_number_edit_pairs':len(real),
                  'split_counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','heldout']},
                  'groups':'Unordered operand pair; disjoint across all splits',
                  'heldout_templates':'Template 3 only, never used in readout training/validation',
                  'numeric_range':'Integers 1 through 20; four arithmetic operations',
                  'probe_selection':'Ridge penalty selected only on validation; final heldout untouched',
                  'test_used':False,'test_note':'Official GSM8K/DART test files remain unused. Synthetic heldout is the final probe evaluation split.',
                  'interpretation':'Information/readout probes, not mathematical solving or original VAE reconstruction',
                  'source_encoder_manifest':str(root/'models/LaDiR-Code-VAE/manifest.json')}
        write(output/'protocol.json',protocol)
        status(stage='loading_encoder',total_texts=len(texts),encoded_texts=0)
        tokenizer=AutoTokenizer.from_pretrained(root/'models/Qwen3-4B-Base',local_files_only=True)
        with torch.device(encoder_device):
            encoder=VAEEncoder(str(root/'models/Qwen3-4B-Base'),64,2560,mask_mode='causal')
        weights=load_file(str(root/'models/LaDiR-Code-VAE/encoder.safetensors'),device='cpu')
        encoder.load_state_dict(weights,strict=True);del weights
        encoder=encoder.float().eval().requires_grad_(False)
        token_rows=[{'input_ids':tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]} for text in texts]
        assert max(len(row['input_ids']) for row in token_rows)<=256,'Unexpected long probe; never truncate'
        flat=[];pooled=[];baseline=[]
        with torch.no_grad():
            for offset in range(0,len(texts),8):
                selected=token_rows[offset:offset+8]
                ids,mask=padded_batch(selected,tokenizer.eos_token_id,encoder_device)
                embeds=encoder.encoder_model.get_input_embeddings()(ids)
                embeds=torch.cat([embeds,encoder.latents_query.expand(len(ids),-1,-1)],dim=1)
                keys=torch.cat([mask,mask.new_ones(len(ids),64)],dim=1)
                hidden=encoder.encoder_model.model(inputs_embeds=embeds,attention_mask=keys,
                       position_ids=positions_from_mask(keys),use_cache=False,return_dict=True).last_hidden_state
                latent=encoder.fc_mu(hidden[:,-64:])
                assert torch.isfinite(latent).all()
                flat.append(latent.flatten(1).cpu())
                pooled.append(torch.cat([latent.mean(1),latent[:,-1]],dim=1).cpu())
                text_hidden=hidden[:,:ids.shape[1]]
                mean=(text_hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                last=text_hidden[torch.arange(len(ids),device=ids.device),mask.sum(1)-1]
                baseline.append(torch.cat([mean,last],dim=1).cpu())
                if offset%64==0 or offset+8>=len(texts):
                    status(stage='encoding',encoded_texts=min(offset+8,len(texts)),
                           progress_percent=100*min(offset+8,len(texts))/len(texts))
        features={'full_latents':torch.cat(flat),'pooled_latents':torch.cat(pooled),'backbone_text':torch.cat(baseline)}
        del flat,pooled,baseline,encoder,hidden,latent
        torch.cuda.empty_cache()
        torch.save({'features':features,'texts':texts},output/'features.pt')
        indices={split:torch.tensor([r['feature_index'] for r in rows if r['split']==split])
                 for split in ['train','validation','heldout']}
        labels=torch.full((len(texts),3),-1,dtype=torch.long)
        for row in rows:labels[row['feature_index']]=torch.tensor([row['a']-1,row['b']-1,OPERATIONS.index(row['operation'])])
        results={};hits={};predictions={}
        for label in ['full_latents','pooled_latents','backbone_text','shuffled_labels']:
            status(stage='fitting_readout',readout=label)
            source='full_latents' if label=='shuffled_labels' else label
            result,prediction,correct,model=ridge_probe(features[source],indices,labels,probe_device,
                                                      shuffle=label=='shuffled_labels')
            results[label]=result;hits[label]=correct;predictions[label]=prediction.tolist()
            torch.save(model,output/f'{label}-readout.pt')
            write(output/'readout-results.json',results)
            print(json.dumps({'readout':label,**result['heldout']}),flush=True)
        paired={name:paired_significance(hits[name],hits['backbone_text']) for name in ['full_latents','pooled_latents']}
        # Query and candidate texts differ; identical self-matches are never used.
        retrieval_results={}
        details=[]
        for mode in ['full_latents','pooled_latents','backbone_text']:
            matrix=features[mode]
            center=matrix[indices['train']].mean(0)
            for centered in [False,True]:
                correct=0;margins=[]
                for row in retrieval:
                    anchor=matrix[row['anchor_index']]
                    candidates=matrix[row['candidate_indices']]
                    if centered:anchor=anchor-center;candidates=candidates-center
                    scores=torch.nn.functional.cosine_similarity(anchor[None],candidates,dim=1)
                    prediction=int(scores.argmax())
                    correct+=prediction==0;margins.append(float(scores[0]-scores[1:].max()))
                    details.append({'feature':mode,'centered':centered,'anchor':row['anchor'],
                                    'candidates':row['candidates'],'scores':scores.tolist(),'predicted':prediction})
                key=mode+('_centered' if centered else '_raw')
                retrieval_results[key]={'correct':correct,'total':len(retrieval),'accuracy':correct/len(retrieval),
                                        'mean_positive_margin':sum(margins)/len(margins),'chance':.25}
        with (output/'retrieval-predictions.jsonl').open('w') as stream:
            for row in details:stream.write(json.dumps(row)+'\n')
        sensitivity=[]
        for row in real:
            first=features['full_latents'][row['original_index']]
            changed=features['full_latents'][row['changed_index']]
            delta=float((first-changed).norm()/first.norm())
            cosine=float(torch.nn.functional.cosine_similarity(first[None],changed[None],dim=1)[0])
            sensitivity.append({**row,'relative_latent_change':delta,'cosine':cosine})
        write(output/'real-number-sensitivity.json',sensitivity)
        heldout_rows=[r for r in rows if r['split']=='heldout']
        with (output/'readout-predictions.jsonl').open('w') as stream:
            for i,row in enumerate(heldout_rows):
                stream.write(json.dumps({**row,'predicted_classes':{k:v[i] for k,v in predictions.items()},
                                         'class_mapping':'first/second operand class +1; operation index in add,subtract,multiply,divide'})+'\n')
        summary={'completed':True,'encoder_weights_trained':False,'protocol':protocol,
                 'readouts':results,'paired_vs_backbone':paired,'paraphrase_retrieval':retrieval_results,
                 'real_numeric_sensitivity':{'pairs':len(sensitivity),
                    'nonidentical_after_edit':sum(row['relative_latent_change']>1e-6 for row in sensitivity),
                    'median_relative_latent_change':float(torch.tensor([r['relative_latent_change'] for r in sensitivity]).median()),
                    'meaning':'Sensitivity only; not proof of numeric recovery or correct arithmetic'},
                 'nominal_uniform_chance':{'operand':.05,'operation':.25,'joint':1/1600},
                 'original_vae_reconstruction_accuracy_available':False,'llada_compatibility_validated':False,
                 'limitations':['Code-domain checkpoint; math encoder from paper not evaluated',
                    'Synthetic probes cover only integers 1–20 and four operations',
                    'Linear readout failure is not proof information is absent',
                    'Linear readout success does not establish text reconstruction or arithmetic-solving ability',
                    'Backbone text pooling is a same-encoder baseline, not a separately trained model',
                    'Full-latent probe has more input dimensions than pooled baselines',
                    'No original VAE decoder bridge is supplied; paper reconstruction cannot be measured directly'],
                 'elapsed_seconds':time.monotonic()-start,
                 'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write(output/'summary.json',summary)
        summary=familiar_wording_diagnostic(output,device=probe_device)
        status(stage='completed',summary=str(output/'summary.json'),encoder_training=False)
        return summary
    except BaseException as error:
        status(stage='failed',error=f'{type(error).__name__}: {error}')
        raise
