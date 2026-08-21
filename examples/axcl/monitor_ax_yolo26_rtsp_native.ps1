[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$AxclSmiPath = "axcl-smi.exe"
)

try {
    $axclSmiCommand = (Get-Command $AxclSmiPath -CommandType Application -ErrorAction Stop).Source
}
catch {
    Write-Error "Cannot find axcl-smi: $AxclSmiPath"
    exit 1
}

while ($true) {
    Clear-Host
    Write-Host ("AX8850 status at {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date))
    & $axclSmiCommand
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Warning "axcl-smi exited with code $exitCode"
    }
    Start-Sleep -Seconds 1
}
