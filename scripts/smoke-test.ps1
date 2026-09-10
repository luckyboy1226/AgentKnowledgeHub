param([ValidateSet('core','full')][string]$Mode='core')
& "$PSScriptRoot/doctor.ps1" -Mode $Mode; if($LASTEXITCODE){exit $LASTEXITCODE}; Invoke-RestMethod 'http://127.0.0.1:8080/api/admin/stats'|ConvertTo-Json -Depth 4
