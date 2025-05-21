# run_nas_pipeline.ps1
# NAS-BNN Full Pipeline Script for CIFAR-10 with Timers

# --- Configuration ---
$DatasetName = "cifar10"
$ArchitectureName = "superbnn_cifar10"
$GlobalWorkers = 0 

$DataPath = "./data/CIFAR10" 
$BaseWorkDir = "./work_dirs/cifar10_nasbnn_exp_run_all_timed" # New dir for timed run
$SupernetCheckpointPath = "$BaseWorkDir/checkpoint.pth.tar"
$SearchOutputDir = "$BaseWorkDir/search"
$SearchInfoFile = "$SearchOutputDir/info.pth.tar"

# Supernet Training Params
$TrainSupernetEpochs = 50
$TrainSupernetBatchSize = 2048
$TrainSupernetLR = "2.5e-3"
$TrainSupernetWD = "5e-6"

# Search Params - IMPORTANT: Review/Update OpsMin/Max after check_ops.py output!
$SearchMaxEpochs = 10
$SearchPopulationNum = 50
$SearchMProb = 0.2
$SearchCrossoverNum = 10
$SearchMutationNum = 10
$SearchOpsMin = 0.03  
$SearchOpsMax = 1.1   
$SearchStep = 0.05
$SearchMaxTrainIters = 10
$SearchTrainBatchSize = 128
$SearchTestBatchSize = 128

# Test Params - IMPORTANT: Update these keys after Search step!
$TestMaxTrainIters = 10
$TestTrainBatchSize = 128
$TestTestBatchSize = 128
$OpsKeyToTest1 = 0    
$OpsKeyToTest2 = 1    

# Fine-tuning Params
$FinetuneBatchSize = 64
$FinetuneLR = "1e-5"
$FinetuneWD = 0
$FinetuneEpochs = 25

# --- Script Execution ---

Write-Host "--------------------------------------------------------------------"
Write-Host "NAS-BNN CIFAR-10 Pipeline Started (with Timers)"
$PipelineStartTime = Get-Date
Write-Host "Overall Start Time: $($PipelineStartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "--------------------------------------------------------------------"

$env:CUDA_VISIBLE_DEVICES="0"
Write-Host "INFO: CUDA_VISIBLE_DEVICES set to: $env:CUDA_VISIBLE_DEVICES"
$env:PYTHONUNBUFFERED="1" 
Write-Host "INFO: PYTHONUNBUFFERED set to: $env:PYTHONUNBUFFERED"
Write-Host "INFO: Current working directory: $(Get-Location)"
Write-Host "INFO: Output for this run will be in: $BaseWorkDir"
Write-Host "---------------------------------"

# Activate Conda/Venv environment (MANUAL STEP - ensure 'nasbnn' env is active before running this .ps1)
# Write-Host "ACTION: Please ensure your 'nasbnn' Python environment is active in this PowerShell session."
# Read-Host -Prompt "Press Enter to continue if environment is active, or Ctrl+C to stop and activate."
# Write-Host "---------------------------------"

# Step 0
$Step0StartTime = Get-Date
Write-Host "Step 0: Preparing CIFAR-10 Data (Started at $(Get-Date -Format 'HH:mm:ss'))..."
python prepare_cifar10.py 
if ($LASTEXITCODE -ne 0) { Write-Error "FATAL: Error in data preparation. Exiting."; exit 1 }
$Step0Duration = New-TimeSpan -Start $Step0StartTime -End (Get-Date)
Write-Host "INFO: CIFAR-10 Data Preparation complete. Duration: $($Step0Duration.ToString()). Data in '$DataPath'"
Write-Host "---------------------------------"

# Step 0.5
Write-Host "Step 0.5: Check Supernet OPs Range (Started at $(Get-Date -Format 'HH:mm:ss'))..."
python check_ops.py 
Write-Host "ACTION: Review OPs range. Current SearchOpsMin = $SearchOpsMin, SearchOpsMax = $SearchOpsMax."
Write-Host "        If these need adjustment, NOW is the time to stop this script (Ctrl+C), edit the variables at the top of this .ps1 file, and restart."
Read-Host -Prompt "Press Enter to continue with current OPs range ($SearchOpsMin - $SearchOpsMax), or Ctrl+C to stop and edit script OPs parameters"
Write-Host "---------------------------------"

# Step 1
$Step1StartTime = Get-Date
Write-Host "Step 1: Training Supernet (Started at $(Get-Date -Format 'HH:mm:ss'))..."
Write-Host "INFO: Supernet training output will be in '$BaseWorkDir'"
if (-not (Test-Path $BaseWorkDir)) { 
    Write-Host "INFO: Creating directory $BaseWorkDir"
    New-Item -ItemType Directory -Force -Path $BaseWorkDir | Out-Null 
}
python train.py `
    --dataset $DatasetName `
    -a $ArchitectureName `
    -b $TrainSupernetBatchSize `
    --lr $TrainSupernetLR `
    --wd $TrainSupernetWD `
    --epochs $TrainSupernetEpochs `
    $DataPath `
    $BaseWorkDir `
    --gpu 0 `
    --workers $GlobalWorkers `
    --print-freq 50 `
    --save-freq 5
if ($LASTEXITCODE -ne 0) { Write-Error "FATAL: Error in Supernet training. Exiting."; exit 1 }
$Step1Duration = New-TimeSpan -Start $Step1StartTime -End (Get-Date)
Write-Host "INFO: Supernet Training complete. Duration: $($Step1Duration.ToString()). Checkpoint: '$SupernetCheckpointPath'"
Write-Host "---------------------------------"

# Step 2
$Step2StartTime = Get-Date
Write-Host "Step 2: Searching Architectures (Started at $(Get-Date -Format 'HH:mm:ss'))..."
Write-Host "INFO: Using SearchOpsMin: $SearchOpsMin, SearchOpsMax: $SearchOpsMax"
Write-Host "INFO: Search output will be in '$SearchOutputDir'"
if (-not (Test-Path $SearchOutputDir)) { 
    Write-Host "INFO: Creating directory $SearchOutputDir"
    New-Item -ItemType Directory -Force -Path $SearchOutputDir | Out-Null 
}
python search.py `
    --dataset $DatasetName `
    -a $ArchitectureName `
    --max-epochs $SearchMaxEpochs `
    --population-num $SearchPopulationNum `
    --m-prob $SearchMProb `
    --crossover-num $SearchCrossoverNum `
    --mutation-num $SearchMutationNum `
    --ops-min $SearchOpsMin `
    --ops-max $SearchOpsMax `
    --step $SearchStep `
    --max-train-iters $SearchMaxTrainIters `
    --train-batch-size $SearchTrainBatchSize `
    --test-batch-size $SearchTestBatchSize `
    --workers $GlobalWorkers `
    $SupernetCheckpointPath `
    $DataPath `
    $SearchOutputDir `
    --gpu 0
if ($LASTEXITCODE -ne 0) { Write-Error "FATAL: Error in Searching. Exiting."; exit 1 }
$Step2Duration = New-TimeSpan -Start $Step2StartTime -End (Get-Date)
Write-Host "INFO: Search complete. Duration: $($Step2Duration.ToString()). Results: '$SearchInfoFile'"
Write-Host "---------------------------------"

# Step 2.5
Write-Host "Step 2.5: Inspect Search Results (Started at $(Get-Date -Format 'HH:mm:ss'))..."
Write-Host "INFO: Search produced '$SearchInfoFile'."
Write-Host "      ACTION: Inspect this file to identify keys for test/finetune (e.g., using your info.pth.tar.py)."
Write-Host "      Python one-liner example: python -c ""import torch; r = torch.load('$SearchInfoFile', map_location='cpu'); print('Pareto Keys:', r.get('pareto_global', {}).keys()); print('Sample Pareto Entry (key 0 if it exists):', r.get('pareto_global',{}).get(0, 'N/A')); print('Sample Pareto Entry (key 1 if it exists):', r.get('pareto_global',{}).get(1, 'N/A'))"" "
Write-Host "INFO: Script will use test/finetune keys: $OpsKeyToTest1 and $OpsKeyToTest2."
Read-Host -Prompt "Press Enter to continue with these keys, or Ctrl+C to stop, edit script, and re-run (from this point if implemented, or earlier steps)"
Write-Host "---------------------------------"

# Step 3a
$Step3aStartTime = Get-Date
Write-Host "Step 3a: Testing Architecture for OPs Key $OpsKeyToTest1 (Started at $(Get-Date -Format 'HH:mm:ss'))..."
$TestOutputDir1 = "$SearchOutputDir/test_ops_key$OpsKeyToTest1"
Write-Host "INFO: Test output for Key $OpsKeyToTest1 will be in '$TestOutputDir1'"
if (-not (Test-Path $TestOutputDir1)) { 
    Write-Host "INFO: Creating directory $TestOutputDir1"
    New-Item -ItemType Directory -Force -Path $TestOutputDir1 | Out-Null 
}
python test.py `
    --dataset $DatasetName `
    -a $ArchitectureName `
    --ops $OpsKeyToTest1 `
    --max-train-iters $TestMaxTrainIters `
    --train-batch-size $TestTrainBatchSize `
    --test-batch-size $TestTestBatchSize `
    --workers $GlobalWorkers `
    $SupernetCheckpointPath `
    $DataPath `
    $SearchInfoFile `
    $TestOutputDir1 `
    --gpu 0
if ($LASTEXITCODE -ne 0) { Write-Warning "WARNING: Error during Test for OPs Key $OpsKeyToTest1." }
$Step3aDuration = New-TimeSpan -Start $Step3aStartTime -End (Get-Date)
Write-Host "INFO: Test for OPs Key $OpsKeyToTest1 Done. Duration: $($Step3aDuration.ToString())"
Write-Host "---------------------------------"

# Step 3b
$Step3bStartTime = Get-Date
Write-Host "Step 3b: Testing Architecture for OPs Key $OpsKeyToTest2 (Started at $(Get-Date -Format 'HH:mm:ss'))..."
$TestOutputDir2 = "$SearchOutputDir/test_ops_key$OpsKeyToTest2"
Write-Host "INFO: Test output for Key $OpsKeyToTest2 will be in '$TestOutputDir2'"
if (-not (Test-Path $TestOutputDir2)) { 
    Write-Host "INFO: Creating directory $TestOutputDir2"
    New-Item -ItemType Directory -Force -Path $TestOutputDir2 | Out-Null 
}
python test.py `
    --dataset $DatasetName `
    -a $ArchitectureName `
    --ops $OpsKeyToTest2 `
    --max-train-iters $TestMaxTrainIters `
    --train-batch-size $TestTrainBatchSize `
    --test-batch-size $TestTestBatchSize `
    --workers $GlobalWorkers `
    $SupernetCheckpointPath `
    $DataPath `
    $SearchInfoFile `
    $TestOutputDir2 `
    --gpu 0
if ($LASTEXITCODE -ne 0) { Write-Warning "WARNING: Error during Test for OPs Key $OpsKeyToTest2." }
$Step3bDuration = New-TimeSpan -Start $Step3bStartTime -End (Get-Date)
Write-Host "INFO: Test for OPs Key $OpsKeyToTest2 Done. Duration: $($Step3bDuration.ToString())"
Write-Host "---------------------------------"

# Step 4a
$Step4aStartTime = Get-Date
Write-Host "Step 4a: Fine-tuning Architecture for OPs Key $OpsKeyToTest1 (Started at $(Get-Date -Format 'HH:mm:ss'))..."
$FinetuneOutputDir1 = "$BaseWorkDir/finetuned_ops_key$OpsKeyToTest1"
Write-Host "INFO: Fine-tuning output for Key $OpsKeyToTest1 will be in '$FinetuneOutputDir1'"
if (-not (Test-Path $FinetuneOutputDir1)) { 
    Write-Host "INFO: Creating directory $FinetuneOutputDir1"
    New-Item -ItemType Directory -Force -Path $FinetuneOutputDir1 | Out-Null 
}
python train_single.py `
    --dataset $DatasetName `
    -a $ArchitectureName `
    -b $FinetuneBatchSize `
    --lr $FinetuneLR `
    --wd $FinetuneWD `
    --epochs $FinetuneEpochs `
    --ops $OpsKeyToTest1 `
    --workers $GlobalWorkers `
    --pretrained $SupernetCheckpointPath `
    $DataPath `
    $SearchInfoFile `
    $FinetuneOutputDir1 `
    --gpu 0
if ($LASTEXITCODE -ne 0) { Write-Warning "WARNING: Error during Fine-tuning for OPs Key $OpsKeyToTest1." }
$Step4aDuration = New-TimeSpan -Start $Step4aStartTime -End (Get-Date)
Write-Host "INFO: Fine-tuning for OPs Key $OpsKeyToTest1 Done. Duration: $($Step4aDuration.ToString())"
Write-Host "---------------------------------"

# Step 4b
$Step4bStartTime = Get-Date
Write-Host "Step 4b: Fine-tuning Architecture for OPs Key $OpsKeyToTest2 (Started at $(Get-Date -Format 'HH:mm:ss'))..."
$FinetuneOutputDir2 = "$BaseWorkDir/finetuned_ops_key$OpsKeyToTest2"
Write-Host "INFO: Fine-tuning output for Key $OpsKeyToTest2 will be in '$FinetuneOutputDir2'"
if (-not (Test-Path $FinetuneOutputDir2)) { 
    Write-Host "INFO: Creating directory $FinetuneOutputDir2"
    New-Item -ItemType Directory -Force -Path $FinetuneOutputDir2 | Out-Null 
}
python train_single.py `
    --dataset $DatasetName `
    -a $ArchitectureName `
    -b $FinetuneBatchSize `
    --lr $FinetuneLR `
    --wd $FinetuneWD `
    --epochs $FinetuneEpochs `
    --ops $OpsKeyToTest2 `
    --workers $GlobalWorkers `
    --pretrained $SupernetCheckpointPath `
    $DataPath `
    $SearchInfoFile `
    $FinetuneOutputDir2 `
    --gpu 0
if ($LASTEXITCODE -ne 0) { Write-Warning "WARNING: Error during Fine-tuning for OPs Key $OpsKeyToTest2." }
$Step4bDuration = New-TimeSpan -Start $Step4bStartTime -End (Get-Date)
Write-Host "INFO: Fine-tuning for OPs Key $OpsKeyToTest2 Done. Duration: $($Step4bDuration.ToString())"
Write-Host "---------------------------------"

$PipelineEndTime = Get-Date
$PipelineTotalDuration = New-TimeSpan -Start $PipelineStartTime -End $PipelineEndTime
Write-Host "--------------------------------------------------------------------"
Write-Host "NAS-BNN CIFAR-10 Pipeline Script Finished Successfully."
Write-Host "Overall End Time: $($PipelineEndTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "Overall Pipeline Duration: $($PipelineTotalDuration.ToString())"
Write-Host "--------------------------------------------------------------------"