Title:  TAU Urban Acoustic Scenes 2020 3Class, Development dataset

# TAU Urban Acoustic Scenes 2020 3Class, Development dataset

[Audio Research Group / Tampere University](http://arg.cs.tut.fi/)

Authors

- Toni Heittola (<toni.heittola@tuni.fi>, <http://www.cs.tut.fi/~heittolt/>)
- Annamaria Mesaros (<annamaria.mesaros@tuni.fi>, <http://www.cs.tut.fi/~mesaros/>)
- Tuomas Virtanen (<tuomas.virtanen@tuni.fi>, <http://www.cs.tut.fi/~tuomasv/>)

Recording and annotation

- Henri Laakso
- Ronal Bejarano Rodriguez
- Toni Heittola

## 1. Dataset

TAU Urban Acoustic Scenes 2020 3Class development dataset consists of 10-seconds audio segments from 10 acoustic scenes grouped into **three major classes** as follows:

- Indoor scenes - ``indoor``: airport, indoor shopping mall, and metro station
- Outdoor scenes - ``outdoor``: pedestrian street, public square, street with medium level of traffic, and urban park
- Transportation related scenes - ``transportation``: travelling by a bus, travelling by a tram, travelling by an underground metro

The dataset contains in total 40 hours of audio.

The dataset was collected by Tampere University of Technology between 05/2018 - 11/2018. The data collection has received funding from the European Research Council under the ERC Grant Agreement 637422 EVERYSOUND.

[![ERC](https://erc.europa.eu/sites/default/files/content/erc_banner-horizontal.jpg "ERC")](https://erc.europa.eu/)

### Preparation of the dataset

The dataset was recorded in 12 large European cities: Amsterdam, Barcelona, Helsinki, Lisbon, London, Lyon, Madrid, Milan, Prague, Paris, Stockholm, and Vienna. For all acoustic scenes, audio was captured in multiple locations: different streets, different parks, different shopping malls. In each location, multiple 2-3 minute long audio recordings were captured in a few slightly different positions (2-4) within the selected location. Collected audio material was cut into segments of 10 seconds length. Acoustic scenes were grouped into **three major classes**: indoor, outdoor, and transportation.

The equipment used for recording consists of a binaural [Soundman OKM II Klassik/studio A3](http://www.soundman.de/en/products/) electret in-ear microphone and a [Zoom F8](https://www.zoom.co.jp/products/handy-recorder/zoom-f8-multitrack-field-recorder) audio recorder using 48 kHz sampling rate and 24 bit resolution. During the recording, the microphones were worn by the recording person in the ears, and head movement was kept to minimum.

Post-processing of the recorded audio involves aspects related to privacy of recorded individuals, and possible errors in the recording process. The material was screened for content, and segments containing close microphone conversation were eliminated. Some interferences from mobile phones are audible, but are considered part of real-world recording process.

A subset of the dataset has been previously published as TUT Urban Acoustic Scenes 2019 Development dataset. Audio segment filenames are retained for the segments coming from this dataset.

### Dataset statistics

The development dataset contains audio material from 10 cities, whereas the evalution dataset (TAU Urban Acoustic Scenes 2020 3Class, evaluation) contains data from all 12 cities. The dataset is perfectly balanced at acoustic scene level, with very slight differences in the number of segments from each city.

#### Audio segments

| Class        | Segments  | Barcelona | Helsinki  | Lisbon   | London   | Lyon     | Milan    | Paris    | Prague   | Stockholm | Vienna   |
| ------------------ | --------- | --------- | --------- | -------- | -------- | ---------| -------- | -------- | -------- | --------- | -------- |
| Indoor             | 4320      | 416       | 437       | 432      | 433      | 432      | 432      | 444      | 432      | 446       | 416      |
| Outdoor            | 5760      | 577       | 577       | 576      | 577      | 576      | 576      | 576      | 576      | 577       | 572      |
| Transportation     | 4320      | 428       | 433       | 432      | 434      | 432      | 432      | 432      | 432      | 433       | 432      |
| **Total**          | **14400** | **1421**  | **1447**  | **1440** | **1444** | **1440** | **1440** | **1452** | **1440** | **1456**  | **1420** |

**Acoustic scene classes**

| Acoustic scene class        | Segments  | Barcelona | Helsinki  | Lisbon   | London   | Lyon     | Milan    | Paris    | Prague   | Stockholm | Vienna   |
| ------------------ | --------- | --------- | --------- | -------- | -------- | ---------| -------- | -------- | -------- | --------- | -------- |
| Airport            | 1440      | 128       | 149       | 144      | 145      | 144      | 144      | 156      | 144      | 158       | 128      |
| Bus                | 1440      | 144       | 144       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| Metro              | 1440      | 141       | 144       | 144      | 146      | 144      | 144      | 144      | 144      | 145       | 144      |
| Metro station      | 1440      | 144       | 144       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| Park               | 1440      | 144       | 144       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| Public square      | 1440      | 144       | 144       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| Shopping mall      | 1440      | 144       | 144       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| Street, pedestrian | 1440      | 145       | 145       | 144      | 145      | 144      | 144      | 144      | 144      | 145       | 140      |
| Street, traffic    | 1440      | 144       | 144       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| Tram               | 1440      | 143       | 145       | 144      | 144      | 144      | 144      | 144      | 144      | 144       | 144      |
| **Total**          | **14400** | **1421**  | **1447**  | **1440** | **1444** | **1440** | **1440** | **1452** | **1440** | **1456**  | **1420** |

#### Recording locations


| Class        | Locations | Barcelona | Helsinki  | Lisbon   | London   | Lyon     | Milan    | Paris    | Prague   | Stockholm | Vienna   |
| ------------------ | --------- | --------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| Indoor             | 133       | 13        | 13        | 12       | 17       | 12       | 11       | 17       | 14       | 13        | 11       |
| Outdoor            | 173       | 19        | 16        | 16       | 17       | 17       | 19       | 17       | 19       | 17        | 16       |
| Transportation     | 208       | 11        | 13        | 28       | 20       | 23       | 26       | 29       | 31       | 15        | 12       |
| **Total**          | **514**   | **43**    | **42**    | **56**   | **54**   | **52**   | **56**   | **63**   | **64**   | **45**    | **39**   |

**Acoustic scene classes**

| Acoustic scene class        | Locations | Barcelona | Helsinki  | Lisbon   | London   | Lyon     | Milan    | Paris    | Prague   | Stockholm | Vienna   |
| ------------------ | --------- | --------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| Airport            | 40        | 4         | 3         | 4        | 3        | 4        | 4        | 4        | 6        | 5         | 3        |
| Bus                | 71        | 4         | 4         | 11       | 7        | 7        | 7        | 11       | 10       | 6         | 4        |
| Metro              | 67        | 3         | 5         | 11       | 4        | 9        | 8        | 9        | 10       | 4         | 4        |
| Metro station      | 57        | 5         | 6         | 4        | 12       | 5        | 4        | 9        | 4        | 4         | 4        |
| Park               | 41        | 4         | 4         | 4        | 4        | 4        | 4        | 4        | 4        | 5         | 4        |
| Public_square      | 43        | 4         | 4         | 4        | 4        | 5        | 4        | 4        | 6        | 4         | 4        |
| Shopping mall      | 36        | 4         | 4         | 4        | 2        | 3        | 3        | 4        | 4        | 4         | 4        |
| Street, pedestrian | 46        | 7         | 4         | 4        | 4        | 4        | 5        | 5        | 5        | 4         | 4        |
| Street, traffic    | 43        | 4         | 4         | 4        | 5        | 4        | 6        | 4        | 4        | 4         | 4        |
| Tram               | 70        | 4         | 4         | 6        | 9        | 7        | 11       | 9        | 11       | 5         | 4        |
| **Total**          | **514**   | **43**    | **42**    | **56**   | **54**   | **52**   | **56**   | **63**   | **64**   | **45**    | **39**   |

### File structure

```
dataset root
│   README.md				this file, markdown-format
│   README.html				this file, html-format
│   meta.csv				meta data, csv-format with a header row, [audio file (string)][tab][class label (string)][tab][identifier (string)][tab][source_label (string)]
│
└───audio					14400 audio segments, 24-bit 48kHz stereo
│   │   airport-barcelona-0-0-a.wav		file naming convention: [scene label]-[city]-[location id]-[segment id]-[device id].wav
│   │   airport-barcelona-0-1-a.wav
│   │   airport-barcelona-0-3-a.wav
│   │   ...
│   │   airport-barcelona-1-17-a.wav
│   │   airport-barcelona-1-18-a.wav
│   │   ...
│
└───evaluation_setup		cross-validation setup, 1 fold
    │   fold1_train.csv		training file list, csv-format with a header row, [audio file (string)][tab][class label (string)]
    │   fold1_test.csv 		testing file list, csv-format with a header row, [audio file (string)]
    │   fold1_evaluate.csv 	evaluation file list, fold1_test.txt with added ground truth, csv-format with a header row, [audio file (string)][tab][class label (string)]

```

## 2. Usage

The partitioning of the data was done based on the location of the original recordings. All segments recorded at the same location were included into a single subset - either **development dataset** or **evaluation dataset**. For each acoustic scene, 1440 segments were included in the development dataset provided here. Evaluation dataset is provided separately.

### Training / test setup

A suggested training/test partitioning of the development set is provided in order to make results reported with this dataset uniform. The partitioning is done such that the segments recorded at the same location are included into the same subset - either training or testing. The partitioning is done aiming for a 70/30 ratio between the number of segments in training and test subsets while taking into account recording locations, and selecting the closest available option. Audio segments coming from nine cities are used for training and all ten cities are used for testing (Milan is used only for testing). Since the dataset includes balanced amount of material from ten cities, this partitioning will leave a small subset of data from Milan unused in the training / test setup. This material can be used when using full dataset to train the system and testing it with evaluation dataset.

The setup is provided with the dataset in the directory `evaluation_setup`.

#### Statistics

| Class        | Train / Segments | Train / Locations | Test / Segments | Test / Locations | Unused / Segments | Unused / Locations |
| ------------------ | ---------------- | ----------------- | --------------- | ---------------- | ----------------- | ------------------ |
| Indoor             | 2704             | 86                | 1297            | 39               | 319               | 8                  |
| Outdoor            | 3757             | 111               | 1604            | 49               | 399               | 13                 |
| Transportation     | 2724             | 128               | 1284            | 61               | 312               | 19                 |
| **Total**          | **9185**         | **325**           | **4185**        | **149**          | **1030**          | **40**             |

**Acoustic scene classes**

| Acoustic scene class        | Train / Segments | Train / Locations | Test / Segments | Test / Locations | Unused / Segments | Unused / Locations |
| ------------------ | ---------------- | ----------------- | --------------- | ---------------- | ----------------- | ------------------ |
| Airport            | 911              | 25                | 421             | 12               | 108               | 3                  |
| Bus                | 928              | 46                | 415             | 20               | 97                | 5                  |
| Metro              | 902              | 41                | 433             | 20               | 105               | 6                  |
| Metro station      | 897              | 37                | 435             | 17               | 108               | 3                  |
| Park               | 946              | 27                | 386             | 11               | 108               | 3                  |
| Public square      | 945              | 28                | 387             | 12               | 108               | 3                  |
| Shopping mall      | 896              | 24                | 441             | 10               | 103               | 2                  |
| Street, pedestrian | 924              | 29                | 429             | 14               | 87                | 3                  |
| Street, traffic    | 942              | 27                | 402             | 12               | 96                | 4                  |
| Tram               | 894              | 41                | 436             | 21               | 110               | 8                  |
| **Total**          | **9185**         | **325**           | **4185**        | **149**          | **1030**          | **40**             |

#### Training

`evaluation setup\fold1_train.csv`
: training file list (in csv-format with a header row)

Format:

    [audio file (string)][tab][scene label (string)]

#### Testing

`evaluation setup\fold1_test.csv`
: testing file list (in csv-format with a header row)

Format:
    [audio file (string)]

#### Evaluating

`evaluation setup\fold1_evaluate.csv`
: evaluation file list (in csv-format with a header row), same as `fold1_test.csv` but with additional reference information. These two files are provided separately to prevent contamination with ground truth when testing the system

Format:

    [audio file (string)][tab][scene label (string)]

### Custom setups

If not using the provided training/test setup, pay attention to the segments recorded at the same location. Location identifier can be found from `meta.csv` or from audio file names:

    [scene label]-[city]-[location id]-[segment id]-[device id].wav

Make sure that all files having **same location id** are placed on the same side of the evaluation. In this dataset, device id is always the same (`a`).

## 3. Changelog

**v1.0 / 2020-02-18**

* Initial commit

## 4. License

License permits free academic usage. Any commercial use is strictly prohibited. For commercial use, contact dataset authors.

    Copyright (c) 2020 Tampere University and its licensors
    All rights reserved.
    Permission is hereby granted, without written agreement and without license or royalty
    fees, to use and copy the TAU Urban Acoustic Scenes 2020 Mobile (“Work”) described in this document
    and composed of audio and metadata. This grant is only for experimental and non-commercial
    purposes, provided that the copyright notice in its entirety appear in all copies of this Work,
    and the original source of this Work, (Audio Research Group at Tampere University of Technology),
    is acknowledged in any publication that reports research using this Work.
    Any commercial use of the Work or any part thereof is strictly prohibited.
    Commercial use include, but is not limited to:
    - selling or reproducing the Work
    - selling or distributing the results or content achieved by use of the Work
    - providing services by using the Work.
    
    IN NO EVENT SHALL TAMPERE UNIVERSITY OR ITS LICENSORS BE LIABLE TO ANY PARTY
    FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE
    OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OR ITS
    LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    TAMPERE UNIVERSITY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
    FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND
    THE TAMPERE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
    UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
