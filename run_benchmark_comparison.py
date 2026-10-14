import os
import sys
import time
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from skimage.metrics import structural_similarity as ssim
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz.distance import Levenshtein

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from networks.d2dewarp_model import D2DewarpModel
from loader.dataset_doc3d_grid_HV import gradient

ocr = RapidOCR()
text_cache = {}

def get_text(img_path):
    if not os.path.exists(img_path):
        return ""
    if img_path in text_cache:
        return text_cache[img_path]
    try:
        img = np.array(Image.open(img_path).convert('RGB'))
        res, _ = ocr(img)
        txt = " ".join([line[1] for line in res]) if res else ""
    except Exception as e:
        print(f"OCR error on {img_path}: {e}", flush=True)
        txt = ""
    text_cache[img_path] = txt
    return txt

def calc_metrics_pair(pred_img_path, gt_img_path):
    text_pred = get_text(pred_img_path)
    text_gt = get_text(gt_img_path)
    
    gt_clean = "".join(text_gt.split())
    pred_clean = "".join(text_pred.split())
    
    ed = Levenshtein.distance(gt_clean, pred_clean)
    cer = ed / max(1, len(gt_clean))
    
    # MS-SSIM
    img_p = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
    img_g = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
    
    if img_p is None or img_g is None:
        score_ssim = 0.0
    else:
        # Resize to standard A4 ratio 595x842 as in DocUNet evaluation
        p_res = cv2.resize(img_p, (595, 842))
        g_res = cv2.resize(img_g, (595, 842))
        score_ssim = float(ssim(g_res, p_res))
        
    return {
        "ssim": round(score_ssim, 4),
        "cer": round(cer, 4),
        "ed": ed,
        "len_pred": len(pred_clean),
        "len_gt": len(gt_clean)
    }

def get_args():
    parser = argparse.ArgumentParser(description="D2Dewarp vs DocTr-Plus Benchmark Comparison")
    parser.add_argument('--model_path', default=os.path.join(BASE_DIR, "model_weight.pt"),
                        help="Path to D2Dewarp model weights")
    parser.add_argument('--crop_dir', default=os.path.join(os.path.dirname(BASE_DIR), "DocTr-Plus", "data benchmark", "crop", "crop"),
                        help="Path to distorted crop images")
    parser.add_argument('--scan_dir', default=os.path.join(os.path.dirname(BASE_DIR), "DocTr-Plus", "data benchmark", "scan", "scan"),
                        help="Path to ground truth scan images")
    parser.add_argument('--doctr_dir', default=os.path.join(os.path.dirname(BASE_DIR), "DocTr-Plus", "results"),
                        help="Path to DocTr results")
    parser.add_argument('--save_path', default=os.path.join(BASE_DIR, "benchmark_results"),
                        help="Path to save benchmark comparison outputs")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Initializing D2Dewarp on {device} ===", flush=True)
    
    input_size = 448
    model_path = args.model_path
    recti_model = D2DewarpModel(img_size=input_size, in_chans=4, hv_out_chans=1, d_model=448)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    sd = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    recti_model.load_state_dict(new_sd)
    recti_model = recti_model.to(device)
    recti_model.eval()
    print("D2Dewarp model loaded successfully!", flush=True)
    
    # Directories
    crop_dir = args.crop_dir
    scan_dir = args.scan_dir
    doctr_dir = args.doctr_dir
    
    save_d2_root = args.save_path
    dewarp_dir = os.path.join(save_d2_root, "dewarp")
    pred_h_dir = os.path.join(save_d2_root, "pred_h")
    pred_v_dir = os.path.join(save_d2_root, "pred_v")
    vis_dir = os.path.join(save_d2_root, "visualizations")
    
    for d in [dewarp_dir, pred_h_dir, pred_v_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)
        
    benchmark_samples = [
        {
            "id": "51_1",
            "crop_file": "51_1 copy.png",
            "scan_file": "51.png",
            "cat": "Dễ - Vừa",
            "desc": "Văn bản hợp đồng / pháp lý chữ 1 cột",
            "b0_file": os.path.join(doctr_dir, "51_1 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "51_1_ill_rec.png")
        },
        {
            "id": "14_1",
            "crop_file": "14_1 copy.png",
            "scan_file": "14.png",
            "cat": "Dễ - Vừa",
            "desc": "Trang sách giáo trình uốn cong nhẹ gáy",
            "b0_file": os.path.join(doctr_dir, "14_1 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "14_1_geo.png")
        },
        {
            "id": "10_1",
            "crop_file": "10_1 copy.png",
            "scan_file": "10.png",
            "cat": "Trung bình",
            "desc": "Sách chữ uốn cong theo gáy chuẩn",
            "b0_file": os.path.join(doctr_dir, "10_1 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "10_1_geo.png")
        },
        {
            "id": "33_1",
            "crop_file": "33_1 copy.png",
            "scan_file": "33.png",
            "cat": "Trung bình",
            "desc": "Trang tài liệu kỹ thuật có viền lề",
            "b0_file": os.path.join(doctr_dir, "33_1 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "33_1_geo.png")
        },
        {
            "id": "48_2",
            "crop_file": "48_2 copy.png",
            "scan_file": "48.png",
            "cat": "Khó",
            "desc": "Bài báo khoa học 2 cột, đồ thị, công thức toán",
            "b0_file": os.path.join(doctr_dir, "48_2 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "48_2_ill_rec.png")
        },
        {
            "id": "25_1",
            "crop_file": "25_1 copy.png",
            "scan_file": "25.png",
            "cat": "Khó - Nếp gấp",
            "desc": "Trang sách bị gấp nếp gãy nhọn (sharp crease)",
            "b0_file": os.path.join(doctr_dir, "25_1 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "25_1_geo.png")
        },
        {
            "id": "64_1",
            "crop_file": "64_1 copy.png",
            "scan_file": "64.png",
            "cat": "Thất bại (Fail)",
            "desc": "Ảnh bị xoay ngược 180 độ trong benchmark gốc",
            "b0_file": os.path.join(doctr_dir, "64_1 copy_thresh.png"),
            "doctr_file": os.path.join(doctr_dir, "64_1_geo.png")
        }
    ]
    
    all_results = []
    
    print("\n" + "=" * 90, flush=True)
    print("STARTING D2DEWARP BENCHMARK INFERENCE & EVALUATION", flush=True)
    print("=" * 90, flush=True)
    
    for item in benchmark_samples:
        sid = item["id"]
        crop_path = os.path.join(crop_dir, item["crop_file"])
        scan_path = os.path.join(scan_dir, item["scan_file"])
        d2_out_path = os.path.join(dewarp_dir, f"{sid}_d2dewarp.png")
        pred_h_path = os.path.join(pred_h_dir, f"{sid}_pred_h.png")
        pred_v_path = os.path.join(pred_v_dir, f"{sid}_pred_v.png")
        
        print(f"\nProcessing [{sid}] ({item['cat']}): {item['desc']}...", flush=True)
        
        # 1. Run D2Dewarp
        img = np.array(Image.open(crop_path).convert("RGB"))
        img_h, img_w, _ = img.shape
        input_img = cv2.resize(img, (input_size, input_size)) / 255.
        
        edge = gradient(img)
        edge = edge[:, :, np.newaxis]
        edge = cv2.resize(edge.astype(np.uint8), (input_size, input_size)) / 255.
        
        with torch.no_grad():
            input_ = torch.from_numpy(input_img).permute(2, 0, 1).to(device).unsqueeze(0).float()
            edge_ = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(device).float()
            
            t0 = time.time()
            pred_h_lst, pred_v_lst, pred_bm = recti_model(torch.cat((input_, edge_), dim=1))
            bm = (2. * (pred_bm / input_size) - 1) * 1.004
            infer_time = time.time() - t0
            
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
        out = F.grid_sample(torch.from_numpy(img / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device), lbl.to(device), align_corners=True)
        img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
        
        cv2.imwrite(d2_out_path, img_geo[:, :, ::-1])
        
        # Save pred_h and pred_v
        pred_h = pred_h_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_h = ((pred_h > 0.2) * 255.).astype(np.uint8)
        cv2.imwrite(pred_h_path, cv2.resize(pred_h, (img_w, img_h)))
        
        pred_v = pred_v_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_v = ((pred_v > 0.2) * 255.).astype(np.uint8)
        cv2.imwrite(pred_v_path, cv2.resize(pred_v, (img_w, img_h)))
        
        print(f"  D2Dewarp inference time: {infer_time:.3f}s. Saved to {d2_out_path}", flush=True)
        
        # 2. Compute metrics for all 4 methods
        # a. Distorted
        m_dist = calc_metrics_pair(crop_path, scan_path)
        # b. Baseline 0 (page-dewarp)
        m_b0 = calc_metrics_pair(item["b0_file"], scan_path) if os.path.exists(item["b0_file"]) else {"ssim": 0, "cer": 1.0, "ed": m_dist["len_gt"], "len_pred": 0, "len_gt": m_dist["len_gt"]}
        # c. Baseline 1 (DocTr)
        m_doctr = calc_metrics_pair(item["doctr_file"], scan_path) if os.path.exists(item["doctr_file"]) else {"ssim": 0, "cer": 1.0, "ed": m_dist["len_gt"], "len_pred": 0, "len_gt": m_dist["len_gt"]}
        # d. D2Dewarp
        m_d2 = calc_metrics_pair(d2_out_path, scan_path)
        
        print(f"  Results for {sid}:", flush=True)
        print(f"    - Distorted : SSIM={m_dist['ssim']:.4f}, CER={m_dist['cer']:.4f}, ED={m_dist['ed']}, Chars={m_dist['len_pred']}/{m_dist['len_gt']}", flush=True)
        print(f"    - Baseline 0: SSIM={m_b0['ssim']:.4f}, CER={m_b0['cer']:.4f}, ED={m_b0['ed']}, Chars={m_b0['len_pred']}/{m_b0['len_gt']}", flush=True)
        print(f"    - DocTr-Plus: SSIM={m_doctr['ssim']:.4f}, CER={m_doctr['cer']:.4f}, ED={m_doctr['ed']}, Chars={m_doctr['len_pred']}/{m_doctr['len_gt']}", flush=True)
        print(f"    - D2Dewarp  : SSIM={m_d2['ssim']:.4f}, CER={m_d2['cer']:.4f}, ED={m_d2['ed']}, Chars={m_d2['len_pred']}/{m_d2['len_gt']}", flush=True)
        
        sample_res = {
            "id": sid,
            "crop_name": item["crop_file"],
            "scan_id": item["scan_file"],
            "category": item["cat"],
            "description": item["desc"],
            "d2_infer_time_sec": round(infer_time, 3),
            "gt_chars": m_dist["len_gt"],
            "distorted": m_dist,
            "baseline_0_pagedewarp": m_b0,
            "baseline_1_doctr": m_doctr,
            "d2dewarp": m_d2
        }
        all_results.append(sample_res)
        
        # 3. Create Multi-Panel Comparison Visualization
        # [Distorted | Baseline 0 | Baseline 1 (DocTr) | D2Dewarp | Ground Truth]
        im_crop = cv2.imread(crop_path)
        im_b0 = cv2.imread(item["b0_file"]) if os.path.exists(item["b0_file"]) else np.zeros((img_h, img_w, 3), dtype=np.uint8)
        im_doctr = cv2.imread(item["doctr_file"]) if os.path.exists(item["doctr_file"]) else np.zeros((img_h, img_w, 3), dtype=np.uint8)
        im_d2 = cv2.imread(d2_out_path)
        im_scan = cv2.imread(scan_path)
        
        target_h = 720
        def resize_h(im):
            if im is None:
                return np.zeros((target_h, int(target_h*0.7), 3), dtype=np.uint8)
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            w = int(im.shape[1] * target_h / im.shape[0])
            return cv2.resize(im, (w, target_h))
            
        r_crop = resize_h(im_crop)
        r_b0 = resize_h(im_b0)
        r_doctr = resize_h(im_doctr)
        r_d2 = resize_h(im_d2)
        r_scan = resize_h(im_scan)
        
        # Add Header banner to each panel
        def add_header(img, title, sub):
            canvas = np.full((img.shape[0] + 65, img.shape[1], 3), 25, dtype=np.uint8)
            cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, sub, (15, 52), cv2.FONT_HERSHEY_DUPLEX, 0.48, (180, 200, 220), 1, cv2.LINE_AA)
            canvas[65:, :] = img
            return canvas
            
        p_crop = add_header(r_crop, "(a) Distorted", f"SSIM: {m_dist['ssim']:.3f} | CER: {m_dist['cer']:.3f}")
        p_b0 = add_header(r_b0, "(b) Baseline 0", f"SSIM: {m_b0['ssim']:.3f} | CER: {m_b0['cer']:.3f}")
        p_doctr = add_header(r_doctr, "(c) DocTr (B1)", f"SSIM: {m_doctr['ssim']:.3f} | CER: {m_doctr['cer']:.3f}")
        p_d2 = add_header(r_d2, "(d) D2Dewarp", f"SSIM: {m_d2['ssim']:.3f} | CER: {m_d2['cer']:.3f}")
        p_scan = add_header(r_scan, "(e) Ground Truth", f"Len: {m_dist['len_gt']} chars")
        
        divider = np.full((p_crop.shape[0], 12, 3), 50, dtype=np.uint8)
        comparison_banner = np.hstack([p_crop, divider, p_b0, divider, p_doctr, divider, p_d2, divider, p_scan])
        
        comp_path = os.path.join(vis_dir, f"comparison_5way_{sid}.png")
        cv2.imwrite(comp_path, comparison_banner)
        print(f"  Saved 5-way visual comparison to {comp_path}", flush=True)
        
        # 4. Create Dual-Dimensions Geometric Feature Visualization
        # [Distorted Input | Pred H Lines | Pred V Lines | Final Dewarped]
        im_ph = cv2.imread(pred_h_path)
        im_pv = cv2.imread(pred_v_path)
        r_ph = resize_h(im_ph)
        r_pv = resize_h(im_pv)
        
        p_ph = add_header(r_ph, "Predicted H-Lines", "Horizontal Dimension")
        p_pv = add_header(r_pv, "Predicted V-Lines", "Vertical Dimension")
        
        hv_banner = np.hstack([p_crop, divider, p_ph, divider, p_pv, divider, p_d2])
        hv_path = os.path.join(vis_dir, f"dual_dimensions_vis_{sid}.png")
        cv2.imwrite(hv_path, hv_banner)
        print(f"  Saved Dual-Dimensions geometric visualization to {hv_path}", flush=True)

    # Save all results to JSON
    json_path = os.path.join(save_d2_root, "benchmark_comparison_full.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nAll benchmark comparison results saved to {json_path}!", flush=True)
    
    # Print summary table
    print("\n" + "=" * 115, flush=True)
    print(f"{'Sample':<8} | {'Category':<14} | {'Method':<22} | {'MS-SSIM':<8} | {'CER':<8} | {'ED':<7} | {'Chars':<12} | {'Time (s)':<8}", flush=True)
    print("-" * 115, flush=True)
    for r in all_results:
        sid = r["id"]
        cat = r["category"]
        gt = r["gt_chars"]
        print(f"{sid:<8} | {cat:<14} | {'Distorted':<22} | {r['distorted']['ssim']:<8.4f} | {r['distorted']['cer']:<8.4f} | {r['distorted']['ed']:<7} | {r['distorted']['len_pred']}/{gt:<8} | {'—':<8}", flush=True)
        print(f"{'':<8} | {'':<14} | {'Baseline 0 (page-dewarp)':<22} | {r['baseline_0_pagedewarp']['ssim']:<8.4f} | {r['baseline_0_pagedewarp']['cer']:<8.4f} | {r['baseline_0_pagedewarp']['ed']:<7} | {r['baseline_0_pagedewarp']['len_pred']}/{gt:<8} | {'~12.5':<8}", flush=True)
        print(f"{'':<8} | {'':<14} | {'Baseline 1 (DocTr)':<22} | {r['baseline_1_doctr']['ssim']:<8.4f} | {r['baseline_1_doctr']['cer']:<8.4f} | {r['baseline_1_doctr']['ed']:<7} | {r['baseline_1_doctr']['len_pred']}/{gt:<8} | {'~2.1':<8}", flush=True)
        print(f"{'':<8} | {'':<14} | {'D2Dewarp (CVPR 2026)':<22} | {r['d2dewarp']['ssim']:<8.4f} | {r['d2dewarp']['cer']:<8.4f} | {r['d2dewarp']['ed']:<7} | {r['d2dewarp']['len_pred']}/{gt:<8} | {r['d2_infer_time_sec']:<8.3f}", flush=True)
        print("-" * 115, flush=True)

if __name__ == '__main__':
    main()
