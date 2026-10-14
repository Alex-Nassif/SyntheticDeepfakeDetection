Installing Dependencies:  (Currently Windows Only)

Run in Root Directory:

Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1

Load Dataset:

python .\model\dataset_loader.py 

Run Model:

python ./model/main.py

Run Interface

python ./interface/app.py

