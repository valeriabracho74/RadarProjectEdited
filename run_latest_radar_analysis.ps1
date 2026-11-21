Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot/.venv/Scripts/python.exe" analyze_radar_3d.py --bin raw_recordings/radar_20261119_232158_393949.bin raw_recordings/radar_20261119_232236_222785.bin raw_recordings/radar_20261119_232329_422620.bin raw_recordings/radar_20261119_232413_874381.bin raw_recordings/radar_20261119_232622_215674.bin --cfg radar_configs/iwr6843_generated.cfg --manifest experiment_20261119.json --range-side negative --no-show
    if ($LASTEXITCODE -ne 0) { throw 'Radar analysis did not finish; see the error above.' }
}
finally { Pop-Location }
