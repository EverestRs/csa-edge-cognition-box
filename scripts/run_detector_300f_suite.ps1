$ErrorActionPreference = 'Stop'
Set-Location "d:/Workspace/Project/VideoAnalyst/campus_security_agent"

$py = "D:/miniconda3/envs/VdAnalyst/python.exe"
$video = "D:/Workspace/Project/VideoAnalyst/data/laying_backward_1-1.mp4"
$outDir = "results/bench300"

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

function Run-Variant {
    param(
        [string]$Name,
        [string[]]$VariantArgs
    )
    Write-Host "Running $Name"
    & $py "scripts/benchmark_detector_single_variant.py" @VariantArgs
}

# Baseline (used for signature and speed comparison)
Run-Variant -Name "baseline" -VariantArgs @(
    "--name", "baseline",
    "--video", $video,
    "--profile", "realtime",
    "--frames", "300",
    "--sample-step", "1",
    "--preprocess-optimized", "true",
    "--graph-opt", "ORT_ENABLE_ALL",
    "--intra-threads", "4",
    "--inter-threads", "1",
    "--enable-mem-pattern", "true",
    "--enable-cpu-arena", "true",
    "--cv-threads", "1",
    "--output", "$outDir/baseline.json"
)

$baselineSig = "$outDir/baseline.json"

# Group 1: equivalent preprocess optimization
Run-Variant -Name "group1_preprocess_off" -VariantArgs @(
    "--name", "group1_preprocess_off",
    "--video", $video,
    "--profile", "realtime",
    "--frames", "300",
    "--sample-step", "1",
    "--preprocess-optimized", "false",
    "--graph-opt", "ORT_ENABLE_ALL",
    "--intra-threads", "4",
    "--inter-threads", "1",
    "--enable-mem-pattern", "true",
    "--enable-cpu-arena", "true",
    "--cv-threads", "1",
    "--baseline-signatures", $baselineSig,
    "--output", "$outDir/group1_preprocess_off.json"
)

Run-Variant -Name "group1_preprocess_on" -VariantArgs @(
    "--name", "group1_preprocess_on",
    "--video", $video,
    "--profile", "realtime",
    "--frames", "300",
    "--sample-step", "1",
    "--preprocess-optimized", "true",
    "--graph-opt", "ORT_ENABLE_ALL",
    "--intra-threads", "4",
    "--inter-threads", "1",
    "--enable-mem-pattern", "true",
    "--enable-cpu-arena", "true",
    "--cv-threads", "1",
    "--baseline-signatures", $baselineSig,
    "--output", "$outDir/group1_preprocess_on.json"
)

# Group 2: quantization model + ORT thread/runtime sweep
$group2 = @(
    @{ n = "group2_int8_t2x1"; onnx = "models/detector/yolov8n_int8.onnx"; intra = "2"; inter = "1"; graph = "ORT_ENABLE_ALL"; mem = "true"; arena = "true" },
    @{ n = "group2_int8_t4x1"; onnx = "models/detector/yolov8n_int8.onnx"; intra = "4"; inter = "1"; graph = "ORT_ENABLE_ALL"; mem = "true"; arena = "true" },
    @{ n = "group2_int8_t6x1"; onnx = "models/detector/yolov8n_int8.onnx"; intra = "6"; inter = "1"; graph = "ORT_ENABLE_ALL"; mem = "true"; arena = "true" },
    @{ n = "group2_fp32n_t4x1"; onnx = "models/detector/yolov8n.onnx"; intra = "4"; inter = "1"; graph = "ORT_ENABLE_ALL"; mem = "true"; arena = "true" },
    @{ n = "group2_fp32n_t6x1"; onnx = "models/detector/yolov8n.onnx"; intra = "6"; inter = "1"; graph = "ORT_ENABLE_ALL"; mem = "true"; arena = "true" },
    @{ n = "group2_int8_t4x1_opt_ext"; onnx = "models/detector/yolov8n_int8.onnx"; intra = "4"; inter = "1"; graph = "ORT_ENABLE_EXTENDED"; mem = "true"; arena = "true" },
    @{ n = "group2_int8_t4x1_no_arena"; onnx = "models/detector/yolov8n_int8.onnx"; intra = "4"; inter = "1"; graph = "ORT_ENABLE_ALL"; mem = "true"; arena = "false" }
)

foreach ($v in $group2) {
    Run-Variant -Name $v.n -VariantArgs @(
        "--name", $v.n,
        "--video", $video,
        "--profile", "realtime",
        "--frames", "300",
        "--sample-step", "1",
        "--onnx-path", $v.onnx,
        "--preprocess-optimized", "true",
        "--graph-opt", $v.graph,
        "--intra-threads", $v.intra,
        "--inter-threads", $v.inter,
        "--enable-mem-pattern", $v.mem,
        "--enable-cpu-arena", $v.arena,
        "--cv-threads", "1",
        "--baseline-signatures", $baselineSig,
        "--output", "$outDir/$($v.n).json"
    )
}

# Group 3: other detector-impact factors
$group3 = @(
    @{ n = "group3_cv_threads_2"; cv = "2"; conf = "-1"; iou = "-1" },
    @{ n = "group3_cv_threads_4"; cv = "4"; conf = "-1"; iou = "-1" },
    @{ n = "group3_conf_040"; cv = "1"; conf = "0.40"; iou = "-1" },
    @{ n = "group3_iou_045"; cv = "1"; conf = "-1"; iou = "0.45" }
)

foreach ($v in $group3) {
    Run-Variant -Name $v.n -VariantArgs @(
        "--name", $v.n,
        "--video", $video,
        "--profile", "realtime",
        "--frames", "300",
        "--sample-step", "1",
        "--preprocess-optimized", "true",
        "--graph-opt", "ORT_ENABLE_ALL",
        "--intra-threads", "4",
        "--inter-threads", "1",
        "--enable-mem-pattern", "true",
        "--enable-cpu-arena", "true",
        "--cv-threads", $v.cv,
        "--confidence-threshold", $v.conf,
        "--iou-threshold", $v.iou,
        "--baseline-signatures", $baselineSig,
        "--output", "$outDir/$($v.n).json"
    )
}

# Build summary table
& $py -c "import json,glob,os; rows=[]; base='results/bench300';
for p in sorted(glob.glob(base+'/*.json')):
 d=json.load(open(p,'r',encoding='utf-8')); rows.append({
  'name':d.get('name',os.path.basename(p)),
  'avg_ms':d.get('avg_ms',0.0),
  'p95_ms':d.get('p95_ms',0.0),
  'fps':d.get('fps',0.0),
  'identical':d.get('output_identical_to_baseline',True),
  'mismatch':d.get('signature_mismatch_frames',0),
  'improve_pct':d.get('improvement_vs_baseline_pct',0.0),
  'sampled_frames':d.get('sampled_frames',0)
 });
out={'video':'$video','frames':300,'rows':rows};
open(base+'/summary.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=2));
print(json.dumps(out,ensure_ascii=False))"

Write-Host "Done: results/bench300/summary.json"
