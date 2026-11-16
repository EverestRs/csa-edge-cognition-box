param(
    [string]$Workspace = "d:\Workspace\Project\VideoAnalyst\campus_security_agent",
    [string]$CondaEnv = "VdAnalyst",
    [string]$RealtimePt = "models/detector/yolov8n.pt",
    [string]$HighAccPt = "models/detector/yolov8x.pt",
    [int]$ImgSize = 640,
    [int]$Opset = 12,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Resolve-WorkspacePath([string]$Base, [string]$Rel) {
    if ([System.IO.Path]::IsPathRooted($Rel)) {
        return $Rel
    }
    return Join-Path $Base $Rel
}

function Copy-IfDifferent([string]$Source, [string]$Destination) {
    $srcFull = [System.IO.Path]::GetFullPath($Source)
    $dstFull = [System.IO.Path]::GetFullPath($Destination)

    if ($srcFull.TrimEnd('\\').ToLowerInvariant() -eq $dstFull.TrimEnd('\\').ToLowerInvariant()) {
        Write-Host "Skip copy (same path): $srcFull"
        return
    }

    Copy-Item -Force $srcFull $dstFull
}

Write-Host "[1/6] Enter workspace: $Workspace"
Set-Location $Workspace

Write-Host "[2/6] Ensure output directories exist"
New-Item -ItemType Directory -Force -Path "models/detector" | Out-Null

if (-not $SkipInstall) {
    Write-Host "[3/6] Install export dependencies in conda env: $CondaEnv"
    conda run -n $CondaEnv pip install ultralytics onnx onnxruntime
}
else {
    Write-Host "[3/6] Skip dependency install"
}

$RealtimePtAbs = Resolve-WorkspacePath $Workspace $RealtimePt
$HighAccPtAbs = Resolve-WorkspacePath $Workspace $HighAccPt

if (-not (Test-Path $RealtimePtAbs)) {
    throw "Realtime pt not found: $RealtimePtAbs"
}
if (-not (Test-Path $HighAccPtAbs)) {
    throw "High-accuracy pt not found: $HighAccPtAbs"
}

Write-Host "[4/6] Export realtime detector ONNX from $RealtimePtAbs"
conda run -n $CondaEnv yolo export model="$RealtimePtAbs" format=onnx imgsz=$ImgSize opset=$Opset simplify=True

Write-Host "[5/6] Export high-accuracy detector ONNX from $HighAccPtAbs"
conda run -n $CondaEnv yolo export model="$HighAccPtAbs" format=onnx imgsz=$ImgSize opset=$Opset simplify=True

$RealtimeOnnxSrc = [System.IO.Path]::ChangeExtension($RealtimePtAbs, ".onnx")
$HighAccOnnxSrc = [System.IO.Path]::ChangeExtension($HighAccPtAbs, ".onnx")

if (-not (Test-Path $RealtimeOnnxSrc)) {
    throw "Export failed, ONNX not found: $RealtimeOnnxSrc"
}
if (-not (Test-Path $HighAccOnnxSrc)) {
    throw "Export failed, ONNX not found: $HighAccOnnxSrc"
}

$RealtimeOnnxDst = Join-Path $Workspace "models/detector/yolov8n_int8.onnx"
$HighAccOnnxDst = Join-Path $Workspace "models/detector/yolov8x.onnx"

Write-Host "[6/6] Rename/Copy ONNX to project target paths"
Copy-IfDifferent $RealtimeOnnxSrc $RealtimeOnnxDst
Copy-IfDifferent $HighAccOnnxSrc $HighAccOnnxDst

Write-Host "Done. Generated files:"
Write-Host " - $RealtimeOnnxDst"
Write-Host " - $HighAccOnnxDst"

Write-Host "Run preflight checks:"
Write-Host " - conda run -n $CondaEnv python tools/preflight_check.py --profile realtime"
Write-Host " - conda run -n $CondaEnv python tools/preflight_check.py --profile high_accuracy"
