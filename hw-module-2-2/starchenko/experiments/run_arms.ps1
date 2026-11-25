# Runs the three prefix arms through the unchanged starter (Windows PowerShell 5.1).
# Paths are relative to this script: PS 5.1 reads BOM-less scripts as ANSI, so no
# non-ASCII path literals are used here.
$h8 = Split-Path -Parent $PSScriptRoot
$data = Join-Path (Split-Path -Parent $h8) "RecSys-HW7\data\FAR-Trans\FAR-Trans"
$env:PREFIX_DIR = Join-Path $h8 "outputs\inputs"
$timing = [ordered]@{}
foreach ($arm in "recency", "lr", "input_order", "llm") {
    $env:PREFIX_ARM = $arm
    $out = Join-Path $h8 "outputs\change_$arm"
    New-Item -ItemType Directory -Force $out | Out-Null
    $t = Measure-Command {
        python (Join-Path $h8 "starter\starter.py") run --data-dir $data --output-dir $out `
            --plugin (Join-Path $h8 "experiments\prefix_plugin.py") --candidate-count 767 |
            Out-File -Encoding utf8 (Join-Path $out "console.txt")
    }
    $timing[$arm] = [math]::Round($t.TotalSeconds, 1)
}
$timing | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $h8 "outputs\starter_wallclock_seconds.json")
$timing
