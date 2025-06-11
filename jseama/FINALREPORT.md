### FINAL REPORT
Problem: With CNN’s, generating data obtained from invasive techniques using only non-invasive methods from lung cancer.
By: Jacob Seaman


Warning git is very poorly managed lololol

PAPER: 
Bakr, S., Gevaert, O., Echegaray, S. et al. A radiogenomic dataset of non-small cell lung cancer. Sci Data 5, 180202 (2018). https://doi.org/10.1038/sdata.2018.202

DATA:
RNA:
Gene Expression Omnibus GSE103584 (2018)

IMAGES:
Bakr, S. et al. The Cancer Imaging Archive https://doi.org/10.7937/K9/TCIA.2017.7hs46erv (2017)


### Data preprocessing
Summary: data preprocessing can be found in .\rna_preprocessing. This ultimately creates a table ./data/df_zscore.csv which is used in other notebooks, where I explore different methodologies.

In total there was a 131 possible data points to study.

The decisions i made for the default was filtering out rna where 80% or less of the data points were nan. This is because I wanted to focus on the genes where i had a complete dataset, less impacted by noise.

I then decided to take the variance and then grabbed the top 100 genes with the highest amount of variance within the sample. This decision was made in order to decrease the sample to a reasonable size and also variance was used as a metric for meaningful. Genes which varied throughout the sample were predicted to be more meaningful then genes which did not.

I took the 10 based log of the data in order to combat skew. Unfortunately as we will discuss later there was still a large skew within the data making it difficult to address or build off of anything other than binary classification

I then calculated the z score for each data point, by taking the mean of each gene and the z score corresponding to the value of each patient. the resultant table is located at ./data/df_zscore.csv, permanently cached and in the git to avoid additional computational costs.


### training
all data was trained using the train method located in model.py. Importantly training often had a patience of 2 making graphs for epochs for different models long or short depending on the patience exhibited. In addition val and train losses were kept track of to properly graph the training in matplot, refer to jupyter notebooks to see an example.


### models
Overall the decision was made to have data pipeline of taking the 8 central slides of an image for each body part of an image. And then each image was put into an individual CNN model, and the training output was purely the mean of all the training output of all the images. This would result in variable image amounts where some patients had more images than others, but the fact the mean was taken was meant to combat image saturation and differences in images. However it is fundamentally limited, it is not unexpected that regression and other more complex models didnt predict with very high accuracy, but for the sake of exploratory research the models breifity and lack of bias is helpful

all in model.py

### Regression Models

class MedicalImageCNN

this is the helper model which took a single image and put it through a **regression** CNN. Look at the model inside for specific parameters used. No fine tuning was used for this model. The output is literally all the possible genes.

class CNNToRNA 

This model was the main regression model. It takes a batch of images and as previously described takes the mean. This outputs a prediction for all 100 of the genes.

# data generation

in helper.py

class PatientDicomDataset

this data set loads all regression, look at the file hierarchy to understand what its doing but as described its taking the 8 central images and skipping the rest so my gpu doesnt die. Main differences from the other dataset methods are the genes which the output for each dataset is an array of genes, other datasets will have a specific gene by index


# results
Regression on the whole dataset can be found in the ./CNN_non_classification.ipynb

The results of a quick and dirty approach did not look fruitious. Which wasnt unexpected it was like shooting for the moon tbh. 

Test MSE: 0.9468834400177002
Test R² : -0.04722974821925163

But performing worse then mean, and the MSE was about 1 standard deviation which is very very high. 


### regression individual genes

Regression perhaps Could still help us identify relevant genes. This is why i constructed a model which built models from each gene individually and tested them. The work can be seen in ./Identifying_individual_genes_regression.ipynb

# new dataset

class PatientDicomDatasetSingleGeneByIndex

This new dataset only loaded a specific gene based on the index. This minor change was necessary as we needed to extract the gene and ignore the other gene values

I picked out the best gene based on the performance and unfortunately there was hiccup and not all data was extracted, however by taking the most current model and graph based on losses it was easy to intuit that this avenue did not have a high rate of success for identifyin specific genes which can be predicted by computer vision.

Test MSE: 0.1547759473323822
Test R² : -9.104867935180664

### Classification

The decision was made to then progress to classification, seeking values which had a large standard deviation, identifying over and under expressed genes was an idea that I thought could be useful.

So classification was created by using thresholds of over 1 std to be 2, 1 to be between -1,1 and under -1 std was thought to be 0. This classification was done on the data generation set but new models had to be built to better utilize cross entropy and have less output layers as well.

# models
class CNNClassifier

This model is a very simple CNN, this is the structure:
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
The idea was to use a very simple model and potentially worry about fine tuning later. we only need to classify for 3 different outcomes

# data-set

class PatientDicomDatasetSingleGeneByIndexClass

This class used the classifier to cast different rna values. This is very similar to the regression dataset generator, only different cause of the casting. but as mentioned this was used:
def zscore_to_class(z, pos_thresh=1.0, neg_thresh=-1.0):
    """
    Convert z-score to classification label based
    the thresholds for overexpression and underexpression are about the same"""
    if z >= pos_thresh:
        return 2  # overexpressed
    elif z <= neg_thresh:
        return 0  # underexpressed
    else:
        return 1  # normal

# results

data exploration can be found at ./classification_identifying_individual_genes.ipynb and ./classification2_identifying_individual.ipynb

unfortunately it was hard to interpret the results meaningfully. The top 4 genes had a 100% accuracy , and loss and val loss was 0... i reran the algorithm in ./classification2_identifying_individual.ipynb to get a better look at the distribution and quickly realized that some of these genes had almost all 2s,0s or 1s. This indicates that data preprocessing needs to be done better to ignore outlier, as most likely its a few outliers which is impacting the z score so heavily for these genes. IT appears not everything follows a normal distribution.

### binary classification

However the data may still be scavengible, if we instead focused on predicting positive and negative values then we can get enough variety in our test and validation set to perform meaningful training and results

# model

The model remains the same, was made mutable to handle different num of classes.

# dataset

class PatientDicomDatasetSingleGeneByIndexClassBinary

this is the same as PatientDicomDatasetSingleGeneByIndexClass but uses 
def zscore_to_class_binary(z, pos_thresh=1.0, neg_thresh=-1.0):
    """
    Convert z-score to classification label based
    the thresholds for overexpression and underexpression are about the same"""
    if z >= 0:
        return 1  # overexpressed
    elif z < 0:
        return 0  # underexpressed

potentially could have reused the prior one but i was doing this at 1 am mind you.

# results
data exploration can be found at ./binary_classification.ipynb

The results were very good for this attempt. The output can be seen below:
gene,auc,accuracy
ORAI2 0.2984073036595395 0.7777777777777779
DNAL1 0.3143503289473684 0.9411764705882353
CHD7 0.4294240851151316 0.96875
SEC22A 0.4331247430098684 1.0
RCOR1 0.4368832236842105 0.703125
CASP2 0.45602256373355265 0.9215686274509803
SAV1 0.4601187455026727 0.4166666666666667
RAB11FIP4 0.49955588892886515 0.6470588235294118
PGS1 0.5249730160361842 0.5119047619047619
ATAD3C 0.5324211120605469 0.8333333333333333
SNORA54 0.5655593872070312 0.5499999999999999
ANO9 0.5947779605263158 0.6309523809523809
DNM1L 0.6033260947779605 0.5066666666666667
ZNF121 0.6186009457236842 0.5714285714285714

Specifically i was able to find that the model was able to predict these genes very well with low auc meaning my model has high confidence. I was able to find statistically meaningful data purely through a CNN, although is this practically meaningful?

### ORAI2 might be practically meaningful

There is evidence that ORAI2 is expressed in various cancers, including in lung cancer. The Human Protein Atlas lists ORAI2 protein expression in lung cancer tissue, suggesting it is present in this cancer type.
https://www.proteinatlas.org/ENSG00000160991-ORAI2
 
Molecular process: Calcium channel, Ion channel
Biological process: Calcium transport, Ion transport, Transport

Specifically recent work has shown that over expression of this gene occurs in cancer. However it has not yet been linked to specific cancer processes.

ORAI1, ORAI2 and ORAI3 to form hexameric CRAC channels. These genes that regulate these channels have been associated with cancer, but the role they play in a cancerous cell is currently being explored.

Vital gene:
“The majority of cellular activities, including but not limited to cell proliferation [1], migration [2], transformation [3], and mitophagy [4], utilize Ca2+ as a second messenger…Specifically ORAI2 and ORAI3 play crucial roles in mediating low range and midrange Ca2+”( Zhang Q, Wang C, He L,2024)

ORAI2 Cancer related research:
“ORAI 2 linked to peritoneal metastasis of gastric cancer cells in gastric cancer”( Zhang Q, Wang C, He L,2024)
Summary, this gene is heavily involved in the AMPK/E2F1/NSUN2/ORAI2 pathway. This pathway increases the spread of cancer along the lining of your stomach.
“ORAI2 overexpression correlates with a reduced risk of systemic recurrence following radical prostatectomy”( Zhang Q, Wang C, He L,2024)
When taking prostate cancer out of someone, ORAI2 overexpression made it less likely for the patient to have it again.

How this could be used in treatment:
“Hasna et al. found that the overexpression of ORAI3 in BC led to a decrease in cell mortality and apoptosis, and an increase in resistance to apoptosis inducers and chemotherapeutic drugs.” ( Zhang Q, Wang C, He L,2024)
If we identify gene levels for a neighbor of ORAI1 influences the efficacy of different treatments for patients.
ORAI1 is linked to experiencing greater amounts of pain for patients with oral cancer. (Ga-Yeon Son et al.,2023)

References from limited literarture review:

Zhang Q, Wang C, He L. ORAI Ca2+ Channels in Cancers and Therapeutic Interventions. Biomolecules. 2024 Mar 29;14(4):417. doi: 10.3390/biom14040417. PMID: 38672434; PMCID: PMC11048467.

Ga-Yeon Son et al.,The Ca2+ channel ORAI1 is a regulator of oral cancer growth and nociceptive pain.Sci. Signal.16,eadf9535(2023).DOI:10.1126/scisignal.adf9535


### FUTURE RESEARCH:
1. Need to redo my preprocessing. (things are more skewed than I would like)
2. Identify medical images highly impactful for the model on ORAI2
Leave one out approach would be relevant here due to the small sample size (long training time cause each sample has multiple images)
3. Then take annotations from images and identify physical characteristics ORAI2 might be heavily involved in which types of lung cancer?
4. Do a deeper literature review and try to find stuff about lungs
5. Collaborate with a biomedical scientist








