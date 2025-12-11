# Setup script para configurar Task Scheduler en Windows
# Ejecutar como Administrador

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "🤖 CONFIGURACIÓN DE TASK SCHEDULER - TENNIS ML" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Obtener directorio del proyecto
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"

Write-Host "📁 Directorio del proyecto: $ProjectDir" -ForegroundColor Yellow
Write-Host "🐍 Python del venv: $VenvPython" -ForegroundColor Yellow
Write-Host ""

# Verificar que existe el venv
if (-not (Test-Path $VenvPython)) {
    Write-Host "❌ Error: No se encontró el entorno virtual en $VenvPython" -ForegroundColor Red
    Write-Host "   Por favor, crea el entorno virtual primero con: python -m venv venv" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Entorno virtual encontrado" -ForegroundColor Green
Write-Host ""

# Función para crear tarea programada
function Create-ScheduledTask {
    param(
        [string]$TaskName,
        [string]$Description,
        [string]$ScriptPath,
        [string]$Time,
        [string]$DaysOfWeek = $null
    )
    
    Write-Host "📋 Creando tarea: $TaskName" -ForegroundColor Cyan
    
    # Crear acción
    $Action = New-ScheduledTaskAction -Execute $VenvPython -Argument $ScriptPath -WorkingDirectory $ProjectDir
    
    # Crear trigger
    if ($DaysOfWeek) {
        $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DaysOfWeek -At $Time
    } else {
        $Trigger = New-ScheduledTaskTrigger -Daily -At $Time
    }
    
    # Configuración
    $Settings = New-ScheduledTaskSettings -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    # Registrar tarea
    try {
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description -Force | Out-Null
        Write-Host "   ✅ Tarea '$TaskName' creada correctamente" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "   ❌ Error creando tarea '$TaskName': $_" -ForegroundColor Red
        return $false
    }
}

Write-Host "📋 Tareas a configurar:" -ForegroundColor Yellow
Write-Host "  - 03:00 AM: Actualización de datos (diaria)" -ForegroundColor White
Write-Host "  - 04:00 AM: Verificación de reentrenamiento (diaria)" -ForegroundColor White
Write-Host "  - 09:00 AM: Predicciones diarias (diaria)" -ForegroundColor White
Write-Host "  - 12:00 PM: Monitoreo del sistema (diaria)" -ForegroundColor White
Write-Host ""

# Preguntar confirmación
$Confirmation = Read-Host "¿Deseas instalar estas tareas programadas? (S/N)"

if ($Confirmation -eq 'S' -or $Confirmation -eq 's') {
    
    Write-Host ""
    Write-Host "🔧 Instalando tareas programadas..." -ForegroundColor Cyan
    Write-Host ""
    
    # 1. Actualización de datos (3:00 AM diaria)
    $Success1 = Create-ScheduledTask `
        -TaskName "TennisML_DataUpdate" `
        -Description "Actualización automática de datos de tenis" `
        -ScriptPath "src\automation\data_updater.py" `
        -Time "03:00"
    
    # 2. Reentrenamiento del modelo (4:00 AM diaria)
    $Success2 = Create-ScheduledTask `
        -TaskName "TennisML_ModelRetrain" `
        -Description "Verificación y reentrenamiento del modelo" `
        -ScriptPath "src\automation\model_retrainer.py" `
        -Time "04:00"
    
    # 3. Predicciones diarias (9:00 AM)
    $Success3 = Create-ScheduledTask `
        -TaskName "TennisML_DailyPredict" `
        -Description "Generación de predicciones diarias" `
        -ScriptPath "src\automation\daily_predictor.py" `
        -Time "09:00"
    
    # 4. Monitoreo del sistema (12:00 PM)
    $Success4 = Create-ScheduledTask `
        -TaskName "TennisML_Monitoring" `
        -Description "Monitoreo del sistema" `
        -ScriptPath "src\automation\monitoring.py" `
        -Time "12:00"
    
    Write-Host ""
    
    if ($Success1 -and $Success2 -and $Success3 -and $Success4) {
        Write-Host "✅ Todas las tareas instaladas correctamente" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Para ver las tareas instaladas:" -ForegroundColor Yellow
        Write-Host "   Get-ScheduledTask | Where-Object {`$_.TaskName -like 'TennisML_*'}" -ForegroundColor White
        Write-Host ""
        Write-Host "📋 Para ejecutar una tarea manualmente:" -ForegroundColor Yellow
        Write-Host "   Start-ScheduledTask -TaskName 'TennisML_DailyPredict'" -ForegroundColor White
        Write-Host ""
        Write-Host "📋 Para deshabilitar una tarea:" -ForegroundColor Yellow
        Write-Host "   Disable-ScheduledTask -TaskName 'TennisML_DailyPredict'" -ForegroundColor White
        Write-Host ""
        Write-Host "📋 Para eliminar todas las tareas:" -ForegroundColor Yellow
        Write-Host "   Get-ScheduledTask | Where-Object {`$_.TaskName -like 'TennisML_*'} | Unregister-ScheduledTask -Confirm:`$false" -ForegroundColor White
    } else {
        Write-Host "⚠️  Algunas tareas no se pudieron instalar" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "❌ Instalación cancelada" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Si quieres instalar manualmente, usa el Task Scheduler de Windows" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "✅ CONFIGURACIÓN COMPLETADA" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
