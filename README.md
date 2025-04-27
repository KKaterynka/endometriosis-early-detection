# Еnhancing early detection and mitigating bias in women’s health through causal machine learning

Implementation of the "Еnhancing early detection and mitigating bias in women’s health through causal machine learning" workflow as the part of the undergraduate dissertation conducted by [Kateryna Koval](https://www.linkedin.com/in/kateryna-koval-0a1147245/) under the supervision of [Liubomyr Bregman](https://www.linkedin.com/in/lbregman/) and with advisory support from [Tetiana Dymytrashchuk](https://www.linkedin.com/in/tetiana-dymytrashchuk/). The clinians [Nataliia Zhemela](https://nccconline.org/doctors/doctor/zhemela-nataliia-ihorivna) and [Yaroslava Misiura](https://nccconline.org/doctors/doctor/misiura-yaroslava-ostapivna) provided clinical consultations that supported a more accurate understanding of the condition and contributed to the validation of the study findings.

## Abstract

Women's health has historically been underrepresented in medical research, resulting in a significant data gap for conditions that predominantly affect women, such as endometriosis. This lack of data has, in turn, impeded the development of effective diagnostic tools.

The aim of this study is to develop a non-invasive screening tool that estimates the likelihood of endometriosis based solely on symptoms that are available for women at the early stages of their diagnostic journey. Our approach addresses the inherent unpredictability of the condition by incorporating the temporality, frequency and intensity of symptoms. Using self-reported questionnaire data from two groups - women diagnosed with endometriosis and those without a diagnosis - we apply machine learning techniques to build non-invasive symptom-based assessment model. In addition to classical machine learning methods, which focus on outcome prediction, we incorporate two causal modelling approaches - manual and automated bias correction - to better assess the diagnostic tool’s potential and account for confounding biases. Furthermore, to address data limitations in women's health research, we investigate the use of synthetically generated data to enhance model accuracy and generalizability to real-world scenarios. Finally, we identify and evaluate the most predictive symptoms of endometriosis, validating our findings through consultations with clinicians and comparisons with existing literature.

The developed approach achieves a diagnostic accuracy of *at least* 82\%, demonstrating a significant step forward in the development of a non-invasive, symptom-based diagnostic tool. Though still in its early stages, this method demonstrates strong potential for clinical application, and practicing clinicians could benefit from the use of this data-driven assessment tool. The proposed method may serve as a valuable aid when minimally invasive methods, such as ultrasound, fail to detect endometriosis lesions. Alternatively, it could be used as a first-line screening tool to help identify individuals who should be referred to specialized radiologists for a non-surgical endometriosis evaluation.

## Repository Organization

The following utility modules are used to support the experimental flows:

- `experiment_utils`: a folder with a module that defines paths, feature sets and helper functions for analyzing and modeling data related to endometriosis prediction.
- `feature_selection_utils`: a folder with a module that supports multiple ML algorithms, model evaluation metrics, cross-validation and SHAP-based model explainability.
- `preprocessing_utils`: a folder with a module that supports preparation of the dataset for model development.
- `visual_utils`: a folder with a module that contains functions to visualize model performance and feature importance.

The following modules are recommended to be run in a Colab environment due to their computational requirements or environment dependencies:

- `ate_estimation`: a directory containing the notebook [causality_ates_endo.ipynb](https://colab.research.google.com/drive/1SSy3NmiqabCy_9D8wFIC4ct_xh1B5k5c#scrollTo=lV8ywNNVmKLi), which includes the derivation of Average Treatment Effect (ATE) values.
- `synthetic_data`: a directory containing the notebook [endometriosis_synthetic_data_generation.ipynb](https://colab.research.google.com/drive/1DVFFCmSvpkftDJpjBzm3cgwbuZAqhNPc), which focuses on generating synthetic data using TVAE (Tabular Variational Autoencoder) and CTGAN (Conditional Tabular Generative Adversarial Network).
- `tabpfn_exp_data`: a directory with files related to TabPFN model training and evaluation. These experiments are conducted in a separate Colab environment, as TabPFN benefits from GPU acceleration. The notebook [TabPFN_endometriosis_experiment.ipynb](https://colab.research.google.com/drive/1S9i1o-kvCWtUDNY7kDj0AAR88KAaJCEo#scrollTo=zJd6GvvQrwlh) contains all relevant experiments for the TabPFN approach.

The root directory contains a set of Jupyter notebooks (*.ipynb), each dedicated to specific experiments, outlined on the beginning or throughout the corresponding notebook. A high-level summary of experimental notebooks is as follows:

| Notebook Name | Description |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `data_preprocessing.ipynb` | Contains the code for preprocessing the collected dataset. |
| `endometriosis_data_stats.ipynb` | Contains the analysis of the collected dataset. |
| `nc_and_mbc_modelling.ipynb` | Contains the experiments for NC (Non-Causal) and MBC (Manual Bias Correction) modelling approaches using real data only. |
| `nc_and_mbc_modelling_with_synthetic.ipynb` | Contains the experiments for NC and MBC modelling approaches using synthetic data for training purposes. |
| `abc_modelling.ipynb` | Contains experiments for the ABC (Automatic Bias Correction) modeling approach, testing both training on real data only and synthetic data only. |

## Data

The data collection process is outlined in the thesis (see Chapter 5), and the collected dataset is not publicly available. To request access, please contact me (Kateryna Koval) with an explanation of your intended use. If the purpose is deemed appropriate, the authors will share the dataset, excluding open-ended responses to ensure participant anonymity.

Throughout the development process, we integrate synthetic data, which is available under Google Drive folder [endo-thesis-data](https://drive.google.com/drive/u/1/folders/1lGtRx5dKdAaWoAuAgTNFOeFN5T-lwt8z).

## Usage

This section provides instructions for running the experiments to reproduce the findings of the study. 

Please note that experiments involving real data cannot be conducted without access to the collected dataset. If you have contacted the authors and received approval, you should have obtained the collected dataset, create a directory named `real_data` and place the file `womens_health_research_answer.csv` inside it. Refer to the Google Drive folder [endo-thesis-data](https://drive.google.com/drive/u/1/folders/1lGtRx5dKdAaWoAuAgTNFOeFN5T-lwt8z) to explore the mentioned structure.

### Clone the Project Repository

```shell
git clone https://github.com/KKaterynka/endometriosis-early-detection
```

### Install Dependecies

This section lists the core dependencies required to run the project experiments. For additional details and any experiment-specific requirements, please refer to the corresponding experiment notebook.

```shell
pip install pandas
pip install numpy
pip install statsmodels
pip install scikit-learn
pip install xgboost
pip install shap
pip install matplotlib
```

The list of dependencies provided here does not include those required for the notebooks `causality_ates_endo.ipynb`, `endometriosis_synthetic_data_generation.ipynb` and `TabPFN_endometriosis_experiment.ipynb`. As previously mentioned, these notebooks are intended to be run in a Colab environment, where all necessary dependencies are explicitly specified and installed within each notebook.

### Experiments Step-by-Step

The image provides a general overview of the workflow. For a detailed explanation, please refer to Chapter 5 of the thesis.

Please note that when prompted, please follow the provided instructions to navigate to the TabPFN Colab notebook and run the corresponding experiments, ensuring consistency with the procedures applied to the other predictive models.

<p align="center">
  <img width="539" alt="Screenshot 2025-04-23 at 14 40 29" src="https://github.com/user-attachments/assets/55b6e7e5-47f5-4bb6-bb35-d1ed5697f73e" />
</p>

1. The process begins with dataset preparation in `data_preprocessing.ipynb`, where the raw data collected through Google Forms is preprocessed into a format suitable for modeling.

2. After the dataset is successfully processed, the preprocessed dataset `preds_preprocessed_endo_data.csv` will be saved in the `real_data` directory.

3. To review the analysis of the collected data, please run the code in the `endometriosis_data_stats.ipynb` notebook.

4. Next, you can run the NC and MBC modeling approaches, initially using real data only, under `nc_and_mbc_modelling.ipynb`. Please note that, for proper execution, you will need to run the TabPFN experiments concurrently using its corresponding notebook. All necessary files related specifically to TabPFN are located in the `tabpfn_exp_data` directory and can be reused as needed.

5. Please note that the data generation process is conducted under the notebook `endometriosis_synthetic_data_generation.ipynb` in a Colab environment. Follow the steps outlined there, or alternatively, you can use the already generated synthetic data available in the Google Drive folder [endo-thesis-data](https://drive.google.com/drive/u/1/folders/1lGtRx5dKdAaWoAuAgTNFOeFN5T-lwt8z) under `synthetic_data` directory. Please ensure to maintain the same structure and create a folder named `synthetic_data` containing the files with the synthetically generated datasets.

6. After completing the NC and MBC modeling approaches using only real data, you can now proceed with running experiments for NC and MBC modeling using synthetic data for training. These experiments are detailed under `nc_and_mbc_modelling_with_synthetic.ipynb`.

7. Finally, to run experiments involving the ABC modeling approach, please refer to the `abc_modelling.ipynb`. Once again, if the training or evaluation of the TabPFN model is involved, please refer to the corresponding TabPFN notebook.
