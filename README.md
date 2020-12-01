# chemical-compunt-analysis
This repository is a cloud version of the Elemnet, IRnet and SVM model for elemental composition analysis. In side the folder [elemnet](https://github.com/CarlosTheran/chemical-compunt-analysis/tree/main/elemnet) you will found the main files to run the differet algorithms;
 1. Elemnet --> dl_regressors.py
 2. IrNet  ---> dl_regressors_irnet.py
 3. SVM    ---> dl_regressors_svm.py

IMPORTANT: In order to run these scripts you must to clone the full repository [ElemNet](https://github.com/CarlosTheran/ElemNet).Then, replaice the folder elemnet from [ElemNet] by [elemnet](https://github.com/CarlosTheran/chemical-compunt-analysis). Remember to install first the [requerements](requirements.txt) packages. The Python3 version used for the virtual env was 3.6.9.  

Now, once you have finished with the enviroment configuration and the installation of respective packages, you can run the different algoritms (Elemnet, IrNet, and SVM) inside of your [elemnet](https://github.com/CarlosTheran/chemical-compunt-analysis) folder. To run the code in an spark enviroment you must to execute the following command.

`spark-submit --master yarn --deploy-mode client <algoritms.py> --config_file sample/sample-run.config`


<algoritms.py> = dl_regressors.py or dl_regressors_irnet.py or dl_regressors_svm.py.
