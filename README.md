#### Patrick Pfenning

Fibromyalgia Final Project:
---

>**OBJECTIVE:** Use the [Fibromyalgia Abstracts](../data/fibro_abstracts.csv) dataset to generate titles from abstract content. Titles behave as a summary of sorts, so title will serve as testing outputs.

>**DELIVERABLES:** [Video Walkthroughs](../recordings)


## Part I:  _Data Cleanup_

### 1. Index Verification

> After loading [Fibromyalgia Abstracts](../data/fibro_abstracts.csv) to a dataframe, it is wise to find a primary key to use as our index. A primary key is a unique set of values for which all data can be mapped. This makes joining other dataframes with a map to the index easy.

>**Key Choice:** _PMID_

>**Justification:** The below shows that the id is unique. Given that it is an integer key, it is easy to reference as the reader.


```python
import pandas as pd

# download set
fibro = pd.read_csv('../data/fibro_abstracts.csv')
# is id truly unique?
if fibro.PMID.is_unique:
    fibro.set_index('PMID', inplace=True)
fibro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>pub_date</th>
      <th>source</th>
      <th>abstract</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>Fibromyalgia, Sjogren's &amp; Depression: Linked?</td>
      <td>2020 Apr 21</td>
      <td>Postgrad Med. 2020 Apr 21. doi: 10.1080/003254...</td>
      <td>Health care has become increasingly fragmented...</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>Electrodiagnostic Abnormalities Associated wit...</td>
      <td>2020</td>
      <td>J Pain Res. 2020 Apr 9;13:737-744. doi: 10.214...</td>
      <td>Purpose: Increasing evidence suggests that fib...</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>High levels of cathepsin S and cystatin C in p...</td>
      <td>2020 Apr 19</td>
      <td>Int J Rheum Dis. 2020 Apr 19. doi: 10.1111/175...</td>
      <td>OBJECTIVES: Although the etiopathogenesis of f...</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>Prevalence and overlap of somatic symptom diso...</td>
      <td>2020 Apr 11</td>
      <td>J Psychosom Res. 2020 Apr 11;133:110111. doi: ...</td>
      <td>OBJECTIVE: To study the prevalence and clinica...</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>Psychometric properties of Turkish version of ...</td>
      <td>2020 Apr 16</td>
      <td>Adv Rheumatol. 2020 Apr 16;60(1):22. doi: 10.1...</td>
      <td>BACKGROUND: Fibromyalgia syndrome (FMS) has ad...</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Extract date from pub_date --> (yr, mo, date)

>**_Approach:_**
>>1. Extract from source
>>    - pros: seems more "precise"
>>    - cons: more text to parse (journal, author, doi, ...)
>>2. Extract from pub_date
>>    - pros: already an attempted date with inteded format of "%Y %b %d"
>>    - cons: will need to extrapolate to get missing data
    

>**_Choice:_**
>>__PMID__ seems to be presorted in decending order. This inherently maps to time of entry, thus an __ffill__ extrapolation will give us a logical approximation for the date unit.


```python
from datetime import datetime as dt
import regex as re
from itertools import cycle, islice, zip_longest

# expect YYYY Mon d
dates = fibro['pub_date'].str.split(expand=True).iloc[:, :3]
dates.columns = ['yr', 'mo', 'day']
dates = dates.fillna(method='ffill').set_index(dates.index)
dates.mo = dates.mo.str.slice(0, 3)
### Will join to finished product 
dates.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yr</th>
      <th>mo</th>
      <th>day</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>2020</td>
      <td>Apr</td>
      <td>21</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>2020</td>
      <td>Apr</td>
      <td>21</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>2020</td>
      <td>Apr</td>
      <td>19</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>2020</td>
      <td>Apr</td>
      <td>11</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>2020</td>
      <td>Apr</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



### 3. Get Publication and DOI from Source

>The __source__ column looks to be a conjunction of __(Publication Title) (Date) (DOI).__ Because we already found date, let us extract to others.


```python
source = fibro.source.str.split(r'(\s\d{4}\s)|(doi:)', expand=True)
print(source.head())
source = source[[0, 6]]
source.columns = ['publication', 'DOI']
source.head() # DOI is okay to be left blank as it id a digital key and may not have existed at time of publication
```

                             0       1     2                    3     4     5  \
    PMID                                                                        
    32314938     Postgrad Med.   2020   None             Apr 21.   None  doi:   
    32308473       J Pain Res.   2020   None   Apr 9;13:737-744.   None  doi:   
    32307906  Int J Rheum Dis.   2020   None             Apr 19.   None  doi:   
    32305723  J Psychosom Res.   2020   None  Apr 11;133:110111.   None  doi:   
    32299495    Adv Rheumatol.   2020   None    Apr 16;60(1):22.   None  doi:   
    
                                                    6     7     8     9  
    PMID                                                                 
    32314938           10.1080/00325481.2020.1758426.  None  None  None  
    32308473   10.2147/JPR.S234475. eCollection 2020.  None  None  None  
    32307906                 10.1111/1756-185X.13840.  None  None  None  
    32305723        10.1016/j.jpsychores.2020.110111.  None  None  None  
    32299495               10.1186/s42358-020-0123-3.  None  None  None  





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publication</th>
      <th>DOI</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>Postgrad Med.</td>
      <td>10.1080/00325481.2020.1758426.</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>J Pain Res.</td>
      <td>10.2147/JPR.S234475. eCollection 2020.</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>Int J Rheum Dis.</td>
      <td>10.1111/1756-185X.13840.</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>J Psychosom Res.</td>
      <td>10.1016/j.jpsychores.2020.110111.</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>Adv Rheumatol.</td>
      <td>10.1186/s42358-020-0123-3.</td>
    </tr>
  </tbody>
</table>
</div>



### 4. Clean Corpus

> Remove bad characters from title/abstract:
> - punctuation
> - non-ascii


```python
clean_text = fibro[['titles', 'abstract']].copy()
clean_text.columns = ['title', 'abstract']  # don't love the plural column name
print(clean_text.head())
for col in clean_text:
    clean_text[col] = clean_text[col].str.encode('ascii', 'ignore') \
                                     .str.decode('ascii') \
                                     .replace(r'(?:[^\w\s]|_)+', '', regex=True) \
                                     .str.strip()
clean_text.head()
```

                                                          title  \
    PMID                                                          
    32314938      Fibromyalgia, Sjogren's & Depression: Linked?   
    32308473  Electrodiagnostic Abnormalities Associated wit...   
    32307906  High levels of cathepsin S and cystatin C in p...   
    32305723  Prevalence and overlap of somatic symptom diso...   
    32299495  Psychometric properties of Turkish version of ...   
    
                                                       abstract  
    PMID                                                         
    32314938  Health care has become increasingly fragmented...  
    32308473  Purpose: Increasing evidence suggests that fib...  
    32307906  OBJECTIVES: Although the etiopathogenesis of f...  
    32305723  OBJECTIVE: To study the prevalence and clinica...  
    32299495  BACKGROUND: Fibromyalgia syndrome (FMS) has ad...  





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>abstract</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>Fibromyalgia Sjogrens  Depression Linked</td>
      <td>Health care has become increasingly fragmented...</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>Electrodiagnostic Abnormalities Associated wit...</td>
      <td>Purpose Increasing evidence suggests that fibr...</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>High levels of cathepsin S and cystatin C in p...</td>
      <td>OBJECTIVES Although the etiopathogenesis of fi...</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>Prevalence and overlap of somatic symptom diso...</td>
      <td>OBJECTIVE To study the prevalence and clinical...</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>Psychometric properties of Turkish version of ...</td>
      <td>BACKGROUND Fibromyalgia syndrome FMS has adver...</td>
    </tr>
  </tbody>
</table>
</div>



### 5. Join Cleaned

>Join the above datasets to a finishes product `fibro_cleaned`


```python
# old
fibro.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>pub_date</th>
      <th>source</th>
      <th>abstract</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>Fibromyalgia, Sjogren's &amp; Depression: Linked?</td>
      <td>2020 Apr 21</td>
      <td>Postgrad Med. 2020 Apr 21. doi: 10.1080/003254...</td>
      <td>Health care has become increasingly fragmented...</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>Electrodiagnostic Abnormalities Associated wit...</td>
      <td>2020</td>
      <td>J Pain Res. 2020 Apr 9;13:737-744. doi: 10.214...</td>
      <td>Purpose: Increasing evidence suggests that fib...</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>High levels of cathepsin S and cystatin C in p...</td>
      <td>2020 Apr 19</td>
      <td>Int J Rheum Dis. 2020 Apr 19. doi: 10.1111/175...</td>
      <td>OBJECTIVES: Although the etiopathogenesis of f...</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>Prevalence and overlap of somatic symptom diso...</td>
      <td>2020 Apr 11</td>
      <td>J Psychosom Res. 2020 Apr 11;133:110111. doi: ...</td>
      <td>OBJECTIVE: To study the prevalence and clinica...</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>Psychometric properties of Turkish version of ...</td>
      <td>2020 Apr 16</td>
      <td>Adv Rheumatol. 2020 Apr 16;60(1):22. doi: 10.1...</td>
      <td>BACKGROUND: Fibromyalgia syndrome (FMS) has ad...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# new
fibro_clean = pd.concat([dates, source, clean_text], axis=1)
fibro_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yr</th>
      <th>mo</th>
      <th>day</th>
      <th>publication</th>
      <th>DOI</th>
      <th>title</th>
      <th>abstract</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>2020</td>
      <td>Apr</td>
      <td>21</td>
      <td>Postgrad Med.</td>
      <td>10.1080/00325481.2020.1758426.</td>
      <td>Fibromyalgia Sjogrens  Depression Linked</td>
      <td>Health care has become increasingly fragmented...</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>2020</td>
      <td>Apr</td>
      <td>21</td>
      <td>J Pain Res.</td>
      <td>10.2147/JPR.S234475. eCollection 2020.</td>
      <td>Electrodiagnostic Abnormalities Associated wit...</td>
      <td>Purpose Increasing evidence suggests that fibr...</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>2020</td>
      <td>Apr</td>
      <td>19</td>
      <td>Int J Rheum Dis.</td>
      <td>10.1111/1756-185X.13840.</td>
      <td>High levels of cathepsin S and cystatin C in p...</td>
      <td>OBJECTIVES Although the etiopathogenesis of fi...</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>2020</td>
      <td>Apr</td>
      <td>11</td>
      <td>J Psychosom Res.</td>
      <td>10.1016/j.jpsychores.2020.110111.</td>
      <td>Prevalence and overlap of somatic symptom diso...</td>
      <td>OBJECTIVE To study the prevalence and clinical...</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>2020</td>
      <td>Apr</td>
      <td>16</td>
      <td>Adv Rheumatol.</td>
      <td>10.1186/s42358-020-0123-3.</td>
      <td>Psychometric properties of Turkish version of ...</td>
      <td>BACKGROUND Fibromyalgia syndrome FMS has adver...</td>
    </tr>
  </tbody>
</table>
</div>



## Part II: _Model Setup_
---

>Now that our data is in a cleansed and in an easily queryable state, if is time to analyze the important parts of this document: Title and Abstract. Our goal here is to develop a title from the given abstract. 

> **DISCLAIMER:** I have chosen to use `SimpleT5` as a pretrain solution. I followed this [tutorial](https://colab.research.google.com/drive/1JZ8v9L0w0Ai3WbibTeuvYlytn0uHMP6O?usp=sharing#scrollTo=zWE4rl2vhaLZ) to set my model up. This also requires a pytorch install.

### 1. Formatting for `SimpleT5`

> `SimpleT5` requires a specidied __input__ and __output__ dataframe which we format below as `df`.


```python
# build dataset

RANDOMS = 42

# example text in and out
print("--- Title ---")
print(fibro_clean["title"].iloc[0])
print("--- Abstract ---")
print(fibro_clean["abstract"].iloc[0])

# t5 set
df = fibro_clean[['abstract', 'title']].copy()
# target column names
df.columns = ['source_text', 'target_text']

# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
df['source_text'] = "summarize: " + df['source_text']
df.head()
```

    --- Title ---
    Fibromyalgia Sjogrens  Depression Linked
    --- Abstract ---
    Health care has become increasingly fragmented partly due to advancing medical technology Patients are often managed by various specialty teams when presenting with symptoms that could be manifestations of different diseases Approximately one third of them are referred to specialists at over half for outpatient appointments1 Fatigue pain depression dry mouth headaches and arthralgia are common complaints and frequently require referral to specialist physicians Differential diagnoses include fibromyalgia FM Sjogrens syndrome SS and depression Evaluations involve various subspecialist especially physicians like those practicing pain management rheumatology and psychiatryThresholds for referring vary Patients sometime feel lost in a medical maze Disagreement is frequent between specialties regarding management12 Each discipline has its own diagnostic and treatment protocols and there is little consensus about shared decisionmaking Communication between doctors could improve continuity There are many differences and similarities in the pathophysiology symptomatology diagnosis and treatment of fibromyalgia Sjogrens syndrome and depression Understanding the associations between fibromyalgia Sjogrens syndrome and depression should improve clinical outcome via a common holistic approach





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_text</th>
      <th>target_text</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32314938</th>
      <td>summarize: Health care has become increasingly...</td>
      <td>Fibromyalgia Sjogrens  Depression Linked</td>
    </tr>
    <tr>
      <th>32308473</th>
      <td>summarize: Purpose Increasing evidence suggest...</td>
      <td>Electrodiagnostic Abnormalities Associated wit...</td>
    </tr>
    <tr>
      <th>32307906</th>
      <td>summarize: OBJECTIVES Although the etiopathoge...</td>
      <td>High levels of cathepsin S and cystatin C in p...</td>
    </tr>
    <tr>
      <th>32305723</th>
      <td>summarize: OBJECTIVE To study the prevalence a...</td>
      <td>Prevalence and overlap of somatic symptom diso...</td>
    </tr>
    <tr>
      <th>32299495</th>
      <td>summarize: BACKGROUND Fibromyalgia syndrome FM...</td>
      <td>Psychometric properties of Turkish version of ...</td>
    </tr>
  </tbody>
</table>
</div>



### 2. Train-Test Split.

> With only having one set of data, it is a good idea to split into train and test subframes. We create an 80% train vs 20% test split below using `sklearn`. This also chooses `PMID` at random so our data sets should choose articles uniformly over time.


```python
# train test split

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2)
train_df.shape, test_df.shape
```




    ((4688, 2), (1173, 2))



## Part III: Model Training

### 1. Initialize

> In this step we will use `train_df` to train our model, as well as evaluate our progress using the first 100 rows of `test_df`. The `SimpleT5` is a pre-trained model tailored for __NLP__ tasks. Additionally, this module will create checkpoint and output structure for each epoch, only keeping the best output.  


```python
from simplet5 import SimpleT5
import os

model = SimpleT5()
model_saved = 'outputs/simplet5-epoch-0-train-loss-1.5917-val-loss-1.4299/'

if os.listdir(model_saved):
    # load trained T5 model
    model.load_model("t5",model_saved, use_gpu=False)
else:
    # load if no checkpoint
    model.from_pretrained(model_type="t5", model_name="t5-base")
    model.train(train_df=train_df[:5000],
                eval_df=test_df[:100], 
                source_max_token_len=128, 
                target_max_token_len=50, 
                batch_size=8, max_epochs=3, use_gpu=False)
```

    Global seed set to 42


### 2 . Predict

> Let's show some predictions for a random delection of the `test_df` which was not used in the evaluation (rows 101 and up) 


```python
# choose 50 rows after the avaluation point
sample_df = test_df[100:].sample(50)
sample_df.head(), sample_df.shape
sample_df.columns = ['abstract', 'title']
```


```python
# print predications using model.predict
sample_df['prediction'] = [model.predict(abstract)[0] for abstract in sample_df.abstract]
```

    Token indices sequence length is longer than the specified maximum sequence length for this model (549 > 512). Running this sequence through the model will result in indexing errors



```python
sample_df[['title', 'prediction']].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>prediction</th>
    </tr>
    <tr>
      <th>PMID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21497140</th>
      <td>Transcranial DC stimulation in fibromyalgia optimized cortical target supported by highresolution computational models</td>
      <td>Effects of transcranial direct current stimulation in fibromyalgia</td>
    </tr>
    <tr>
      <th>28779930</th>
      <td>Effects of whole body vibration therapy in pain function and depression of the patients with fibromyalgia</td>
      <td>Effects of whole body vibration in fibromyalgia patients</td>
    </tr>
    <tr>
      <th>16572917</th>
      <td>Eosinophilic perifolliculitis presenting as a painful cystic ovarian mass in a woman with fibromyalgia a case report</td>
      <td>Eosinophilic perifolliculitis in women with premenopausal and perimenopausal ovarian failure</td>
    </tr>
    <tr>
      <th>20855168</th>
      <td>Internetenhanced management of fibromyalgia a randomized controlled trial</td>
      <td>Internetbased exercise and behavioral selfmanagement for fibromyalgia</td>
    </tr>
    <tr>
      <th>10953277</th>
      <td>Fibromyalgia following trauma psychology or biology</td>
      <td>The role of biology and psychology in the development of fibromyalgia</td>
    </tr>
    <tr>
      <th>30682223</th>
      <td>Evidence for an association between endometriosis fibromyalgia and autoimmune diseases</td>
      <td>Prevalence of endometriosis and fibromyalgia in women with autoimmune disease</td>
    </tr>
    <tr>
      <th>19327791</th>
      <td>Cognitive profile in fibromyalgia Comparison with a mild cognitive impairment group</td>
      <td>Neuropsychological assessment of patients with fibromyalgia and with subjective cognitive complaints</td>
    </tr>
    <tr>
      <th>12237183</th>
      <td>Enhanced temporal summation of second pain and its central modulation in fibromyalgia patients</td>
      <td>Temporal summation windup and decay of second pain in fibromyalgia patients</td>
    </tr>
    <tr>
      <th>31964420</th>
      <td>Treatment data from the Brazilian fibromyalgia registry EpiFibro</td>
      <td>A crosssectional study of the treatment of patients with fibromyalgia</td>
    </tr>
    <tr>
      <th>26825913</th>
      <td>Dry Eye Syndrome Risks in Patients With Fibromyalgia A National Retrospective Cohort Study</td>
      <td>Fibromyalgia and dry eye syndrome</td>
    </tr>
    <tr>
      <th>26778226</th>
      <td>Shortterm complementary and alternative medicine on quality of life in women with fibromyalgia</td>
      <td>A comparative study of acupuncture electroacupuncture and moxibustion in women with fibromyalgia</td>
    </tr>
    <tr>
      <th>15180033</th>
      <td>Mitochondrial encephalomyopathy with lactic acidosis and strokelike episodes MELAS a mitochondrial disorder presents as fibromyalgia</td>
      <td>Fibromyalgia and mitochondrial disorders</td>
    </tr>
    <tr>
      <th>18222129</th>
      <td>Emotions and emotional approach and avoidance strategies in fibromyalgia</td>
      <td>Emotionalavoidance and emotionalavoidance strategies in women with fibromyalgia</td>
    </tr>
    <tr>
      <th>2735155</th>
      <td>Personality markers defense behavior and illness concept in patients with primary fibromyalgia</td>
      <td>Fibromyalgia and the role of personality traits in disease behavior</td>
    </tr>
    <tr>
      <th>26843364</th>
      <td>Associations between patterns of active commuting and socioeconomic factors in women with fibromyalgia the alAndalus project</td>
      <td>Active commuting behaviour in women with fibromyalgia and controls</td>
    </tr>
    <tr>
      <th>3801074</th>
      <td>The natural history of fibromyalgia</td>
      <td>Natural history of fibromyalgia</td>
    </tr>
    <tr>
      <th>11037075</th>
      <td>Treating fibromyalgia</td>
      <td>Fibromyalgia and the neuroendocrine system</td>
    </tr>
    <tr>
      <th>15283010</th>
      <td>Decreased sleep spindles and spindle activity in midlife women with fibromyalgia and pain</td>
      <td>Spindle frequency activity in midlife women with fibromyalgia and moderate to high pain</td>
    </tr>
    <tr>
      <th>12122925</th>
      <td>Management of dysautonomia in fibromyalgia</td>
      <td>Dysautonomia in the pathogenesis of fibromyalgia</td>
    </tr>
    <tr>
      <th>31131959</th>
      <td>A diagnosis of rheumatoid arthritis endometriosis or IBD is associated with later onset of fibromyalgia and chronic widespread pain</td>
      <td>A longitudinal study of fibromyalgia and chronic widespread pain in patients with rheumatoid arthritis endometriosis or inflammatory bowel disease</td>
    </tr>
  </tbody>
</table>
</div>


