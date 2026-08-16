from pathlib import Path
import json, pickle, struct, zlib
import pandas as pd

import mammography_agent.input_fidelity as inf


def _write_png16(path: Path, values):
    # Minimal valid 16-bit grayscale PNG for IHDR-level contract auditing.
    path.parent.mkdir(parents=True, exist_ok=True)
    height=len(values); width=len(values[0])
    sig=b"\x89PNG\r\n\x1a\n"
    ihdr=struct.pack(">IIBBBBB",width,height,16,0,0,0,0)
    def chunk(kind,data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
    # Include unfiltered scanlines so the file is valid if opened by another PNG reader.
    raw=b"".join(b"\x00"+b"".join(struct.pack(">H",int(v)) for v in row) for row in values)
    path.write_bytes(sig+chunk(b"IHDR",ihdr)+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))


def test_input_fidelity_reads_chunked_preprocessing_metadata_without_inference(tmp_path, monkeypatch):
    root=tmp_path/'workspace'; run=root/'output'/'normal_tests'/'normal-x'; run.mkdir(parents=True)
    imgs=[]
    for name in ['L_CC','R_CC','L_MLO','R_MLO']:
        p=root/'datasets'/'processed'/'cbis_ddsm'/'images'/f'S1_{name}.png'
        _write_png16(p, [[0,1000,2000],[0,3000,4000]])
        imgs.append(p)
    selected=pd.DataFrame([{
        'study_id':'S1','patient_id':'P1','ground_truth':1,'left_ground_truth':1,'right_ground_truth':0,
        'l_cc':str(imgs[0]),'r_cc':str(imgs[1]),'l_mlo':str(imgs[2]),'r_mlo':str(imgs[3]),'horizontal_flip':'NO'
    }])
    selected.to_csv(run/'selected_studies.csv',index=False)
    batch=run/'chunks'/'0000'/'model_batch'; batch.mkdir(parents=True)
    pd.DataFrame({'position':[0],'study_id':['S1'],'study_key':['S1']}).to_csv(batch/'study_order.csv',index=False)
    for model in ['gmic','nyu','glam']:
        pre=batch/'preprocessed'/model/f'nyu_{model}_x_cropped_images'; pre.mkdir(parents=True,exist_ok=True)
        exam={
            'horizontal_flip':'NO',
            'distance_from_starting_side':{'L-CC':[0],'R-CC':[0],'L-MLO':[2],'R-MLO':[0]},
            'best_center':{'L-CC':[(1,1)],'R-CC':[(1,1)],'L-MLO':[(1,1)],'R-MLO':[(1,1)]},
        }
        with (pre/'cropped_exam_list.pkl').open('wb') as fh: pickle.dump([exam],fh,protocol=4)
    monkeypatch.setattr(inf,'DEFAULT_ANALYSES',root/'output'/'analyses')
    source=root/'datasets'/'raw'/'cbis_ddsm'/'source_manifest.csv'
    out=inf.audit_input_fidelity(run, source_manifest=source)
    summary=json.loads((out/'input_fidelity_summary.json').read_text())
    assert summary['input_png_contract']['all_valid'] is True
    assert summary['input_png_contract']['invalid_images'] == 0
    assert summary['upstream_preprocessing_contract']['distance_from_starting_side_nonzero_records'] == 3
    assert summary['research_guards']['model_inference_performed'] is False
    prep=pd.read_csv(out/'model_preprocessing_audit.csv')
    assert len(prep)==12
    assert prep.distance_nonzero.sum()==3


def test_input_fidelity_missing_source_dicom_is_diagnostic_not_failure(tmp_path, monkeypatch):
    root=tmp_path/'workspace'; run=root/'output'/'normal_tests'/'normal-x'; run.mkdir(parents=True)
    vals=[]
    for col in ['l_cc','r_cc','l_mlo','r_mlo']:
        p=root/f'{col}.png'; _write_png16(p,[[0,1],[2,3]]); vals.append(str(p))
    pd.DataFrame([{'study_id':'S1','patient_id':'P1','ground_truth':0,'left_ground_truth':0,'right_ground_truth':0,
                   'l_cc':vals[0],'r_cc':vals[1],'l_mlo':vals[2],'r_mlo':vals[3]}]).to_csv(run/'selected_studies.csv',index=False)
    monkeypatch.setattr(inf,'DEFAULT_ANALYSES',root/'output'/'analyses')
    out=inf.audit_input_fidelity(run,source_manifest=root/'missing.csv')
    d=pd.read_csv(out/'dicom_conversion_audit.csv')
    assert not d.dicom_available.any()
    summary=json.loads((out/'input_fidelity_summary.json').read_text())
    assert summary['dicom_conversion_contract']['dicom_headers_available']==0
