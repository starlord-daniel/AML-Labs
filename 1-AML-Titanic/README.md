# Titanic ML - Classification

This example is based on actual data from the Titanic catastrophe. In this example a classification algorithm is built that is able to predict, if a fictual person would have survived the Titanic accident.

#### Overview?
Anchor Links in MD

## Prerequisits
* Create an Azure Machine Learning Experimentation account.
* Azure Machine Learning Workbench installed

You can follow the instructions in the [Install and create Quickstart](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation) article to install the Azure Machine Learning Workbench application.

## Setup
1. Start of by cloning the [repository](https://github.com/starlord-daniel/AML-Labs) on your local machine.
2. The files needed for the experiment are located under *1-AML-Titanic*
3. Open *Azure Machine Learning Workbench* and add the project to your workspace

![Create Project](assets/Create_Project.PNG )

4. Fill in the **Project name** and **Project directory** boxes. **Project description** is optional but helpful. Leave the Visualstudio.com GIT Repository URL box blank for now and choose a workspace (created in the [installation guide](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation)).

After setting up your project you should see the dashboard of the project.

## Get your Data in place
Preparing the data is one of the key steps for your first ML experiment. This step includes renaming of columns, filtering (f.e. remove *null* values), transform values and change data type.

To start of, add your dataset ([titanic.csv](data/titanic-dataset.csv)) as a new Data Source to the experiment and follow the steps.

![Add Data Source](assets/Add_Data_Source.PNG )

(**Please note:** The project you cloned already has a fully prepared dataset included. However, to get a glimpse of how things work we recommend to try it out on your own.)

### Data Exploration
In the *titanic.csv* file you can find the data used by the model. Before we get started with preparing the data for the experiment let's have a quick look at what the data is about.

| Variable | Definition | Key |
| ------------- |-------------| -----|
| survival | Survival | 0 = no, 1 = yes |
| pclass | Ticket class |   1 = 1st, 2 = 2nd, 3 = 3rd |
| sex | Sex ||
| Age | Age in years ||
| sibsp | # of siblings / spouses aboard the Titanic ||
| parch | # of parents / children aboard the Titanic ||
| ticket | Ticket number ||
| fare | Passenger Fare ||
| cabin | Cabin number ||
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
| boat | # of life boat if survived ||
| body | # of body if dead ||

To get to know your data even better by exploring certain statistical figures click on *Metrics* in the top bar of the data section:

![Data Exploration](assets/Data_Exploration_1.PNG )

![Data Exploration](assets/Data_Exploration_2.PNG )

### Data Preparation
Now it's time to get our hands dirty and prepare our data. The following steps are about getting rid of data we don't need, removing incomplete data and transforming data types, but let's take one step after the other:

* To start of create a new Preparation file

![Prepare Data](assets/Prepare_Data.PNG )

* Select the Prepare button. The Prepare dialog box opens. 
The sample project comes with an titanic-clear.dprep file. By default, it asks you to create a new data flow in the titanic-clear.dprep data preparation package that already exists. 
Select + New Data Preparation Package from the drop-down menu, enter a new value for the package name, use titanic-clear-1, and then select OK.
* Let's start with removing the columns we don't need. Therefore select the column "*name*" and click "remove". (right-click on the column head)
Repeat this for the columns *teicket, embarked, cabin, boat, body* and *home.dest*
* Next we filter the pclass for empty values

![Filter Data](assets/Filter.PNG )

Repeat this for the columns *fare* and *age*
* We also want *sex* to be a numeric value (female = 1, male = 0). To do so, we have to select "Replace Values"

![Replace Data](assets/Replace.PNG )

Repeat this for male
* To make *sex* a numeric value we only have to click on the column header and select the *numeric* option.
* Last but not least we want to change the precision of the columns *age* and *fare*.

The Azure Machine Learning Workbench shows you all the steps you performed in the *Steps* pane on the right side. There you can edit and revert to early stages. Great huh?

![Transform Data](assets/Transformation_Steps.PNG )

Here you can view, change and edit single steps performed during the preparation process.
The result of the above-mentioned steps should be the same as the dataset prepared in the demo.

For more information on how to prepare data in Azure Machine Learning Workbench, see the [Get started with data preparation guide](https://docs.microsoft.com/en-us/azure/machine-learning/preview/data-prep-getting-started).

## Develop the Model
At this stage we have our data sources configured and prepared for the fun part you've probably been waiting for - developing the ML mmodel.
To make things a little bit easier we already added the [train.py](train.py) file to the project. Open the file and try to make yourself familiar with the code.
Basically the script performs the following steps:

* Loads the data preparation package titanic-clear.dprep to create a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).
* Spilts the dataset into a training and a testing set (65%/35%)
* Uses the scikit-learn machine learning library to build a logistic regression model. 
* Serializes the model by inserting the pickle library into a file in the outputs folder. 
* The run_logger object is used throughout to record the model accuracy into the logs. The logs are automatically plotted in the run history.

The deserialized model, saved in the ouputs folder, can later be used to make a prediction on a new record but first things first.

### Run your Python Script
We are now ready to run the script on your local computer. 
Select *local* as the execution target from the command bar near the top of the application, and select *train.py* as the script to run. There are other files included in the sample that we will check out later. 

![Run Script](assets/Run_Script.PNG )

* Click the Run button to begin running train.py on your computer. 
* The Jobs panel slides out from the right if it is not already visible, and an train job is added in the panel. Its status transitions from Submitting to Running as the job begins to run, and then to Completed in a few seconds. 
* **Congratulations.** You have successfully executed a Python script in Azure Machine Learning Workbench.

### Check Results
After running the script 2-3 times feel free to check the results in the Jobs History.

![Script Results](assets/Script_Results.PNG )

* Accuracy shows you how accurate your script works.
* The list of Jobs below show a historical list of the scrips you ran. By selecting one job you get a detailed overview.

![Script Log](assets/Script_Log.PNG )

## Deploy your Model
Before we can think about deploying our experiment we have to figure out, if our model actually works. Therefor, it is important to download and save the model, created by our train.py script. The model.pkl file is saved in the output section of the log.

![Script Log](assets/Script_Log.PNG )

The file is used in our second script - [*score.py*](score.py). This file loads the model.pkl and tests it against a predefined dataframe. The result is a JSON-Object. You can find it in the Jobs history under the score.py section. Selecting a certain Job in the list will take you to a detailed overview (sounds familiar?).

You now have:
* Setup Azure Machine Learning Workbench
* Created a project
* Added a new Data Source
* Prepared the data in order to be able to process it
* Ran your first training scripts and created a model
* Used the model to test your data and classify it
* Deployed your model somewhere?