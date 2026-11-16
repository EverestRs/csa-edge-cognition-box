param(
    [Parameter(Mandatory = $true)]
    [string]$TaskId,
    [int]$Seconds = 20,
    [int]$MaxSamples = 120
)

$wsUrl = "ws://127.0.0.1:8000/ws/task/$TaskId"
Add-Type -AssemblyName System.Net.WebSockets
Add-Type -AssemblyName System.Text

$ws = [System.Net.WebSockets.ClientWebSocket]::new()
$cts = [System.Threading.CancellationTokenSource]::new()
$ws.ConnectAsync([Uri]$wsUrl, $cts.Token).GetAwaiter().GetResult()

$sendBytes = [System.Text.Encoding]::UTF8.GetBytes('start')
$sendSeg = [ArraySegment[byte]]::new($sendBytes)
$ws.SendAsync($sendSeg, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $cts.Token).GetAwaiter().GetResult()

$buffer = New-Object byte[] 131072
$det = 0.0
$pose = 0.0
$action = 0.0
$post = 0.0
$sumStages = 0.0
$pipe = 0.0
$count = 0
$deadline = (Get-Date).AddSeconds($Seconds)

while ($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open -and (Get-Date) -lt $deadline -and $count -lt $MaxSamples) {
    $seg = [ArraySegment[byte]]::new($buffer)
    $res = $ws.ReceiveAsync($seg, $cts.Token).GetAwaiter().GetResult()
    if ($res.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) { break }
    if ($res.Count -le 0) { continue }

    $txt = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $res.Count)
    try {
        $obj = $txt | ConvertFrom-Json -Depth 16
        $stage = $obj.metrics.pipeline_stage_ms
        if ($null -ne $stage) {
            $det += [double]$stage.detector
            $pose += [double]$stage.pose
            $action += [double]$stage.action
            $post += [double]$stage.post
            $sumStages += [double]$stage.sum_stages
            $pipe += [double]$obj.metrics.pipeline_latency_ms
            $count += 1
        }
    } catch {}
}

if ($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
    $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, 'done', $cts.Token).GetAwaiter().GetResult()
}

if ($count -eq 0) {
    [PSCustomObject]@{
        task_id = $TaskId
        samples = 0
        note = 'No pipeline_stage_ms received. Restart backend with latest code.'
    } | ConvertTo-Json -Depth 6
    exit 0
}

$avgDet = $det / $count
$avgPose = $pose / $count
$avgAction = $action / $count
$avgPost = $post / $count
$avgSum = $sumStages / $count
$avgPipe = $pipe / $count

[PSCustomObject]@{
    task_id = $TaskId
    samples = $count
    avg_ms = @{
        detector = [math]::Round($avgDet, 3)
        pose = [math]::Round($avgPose, 3)
        action = [math]::Round($avgAction, 3)
        post = [math]::Round($avgPost, 3)
        sum_stages = [math]::Round($avgSum, 3)
        pipeline_compute = [math]::Round($avgPipe, 3)
    }
    share_pct_of_stage_sum = @{
        detector = [math]::Round((100 * $avgDet / [math]::Max(0.001, $avgSum)), 2)
        pose = [math]::Round((100 * $avgPose / [math]::Max(0.001, $avgSum)), 2)
        action = [math]::Round((100 * $avgAction / [math]::Max(0.001, $avgSum)), 2)
        post = [math]::Round((100 * $avgPost / [math]::Max(0.001, $avgSum)), 2)
    }
    sum_vs_pipeline_pct = [math]::Round((100 * $avgSum / [math]::Max(0.001, $avgPipe)), 2)
} | ConvertTo-Json -Depth 8
