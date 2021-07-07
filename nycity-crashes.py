import pandas as pd


# work on small datasets by changing the iloc slicing
directory = r"C:\Users\Amaury\Documents\Data_Science_BECODE\Projects\Python\GNT-Arai-2.31\content\4.machine_learning\Nycity-crashes\NYC-MotorVC\data_100000.csv"
df_crash_ori = pd.read_csv(directory, sep=',').iloc[0:100000,:]
df_data_100000 = df_crash_ori.copy()

df_data_100000.head()
print(df_data_100000.shape)
print(df_data_100000.info())
# print("----- columns with missing values = True -------->")
# print(df_data_100000.isnull().any())
# print("----------- missing values check END ------------>")

# check types
# print(df_data_100000.dtypes)

def remove_empty_rows(df):
    return df.dropna(how='all')


def change_to_date_type(column):
    df_data_100000[column] = df_data_100000[column].astype("datetime64[ns]")


def write_to_csv(file_path):
    df_data_100000.to_csv(file_path)


# fill empty rows with undefined
def fill_text(column, fill):
    df_data_100000[column] = df_data_100000[column].fillna(value=fill)


# # drop duplicates in the tree_id column
# def drop_duplicates(column):
#     return df_data_100000.drop_duplicates(subset=column, keep=False)

def create_categories(column):
    df_data_100000[column] = df_data_100000[column].astype(dtype="category")


#def change_col_to_category(column):
#     df_data_100000[column] = df_data_100000[column].astype("category")


'''
functions/classes above - calling below
'''

#remove
# remove empty rows, if any
remove_empty_rows(df_data_100000)

# drop the redundant columns
df_data_100000 = df_data_100000.drop(["borough", "location", "number_of_persons_injured", "number_of_persons_killed"], axis=1)

# put the index as collision ID
df_data_100000.set_index("collision_id")

# fix the date time column
df_data_100000["crash_date"] = df_data_100000["crash_date"].str.replace(r'T00:00:00.000', '')

# change into date data type
cols_to_date_type = ["crash_date"]
change_to_date_type(cols_to_date_type)

# delete empty locations
list_empty_locations = ["zip_code", "latitude", "longitude", "on_street_name", "off_street_name", "cross_street_name"]
df_data_100000.dropna(subset=list_empty_locations, how='all')

# fill with undefined
columns_to_be_filled = ["on_street_name", "off_street_name", "cross_street_name"]
fill_text(columns_to_be_filled, "undefined")

# # fill with unspecified
# columns_cf = ["contributing_factor_vehicle_1", "contributing_factor_vehicle_2", "contributing_factor_vehicle_3", "contributing_factor_vehicle_4", "contributing_factor_vehicle_5"]
# fill_text(columns_to_be_unspecified, "unspecified")

# # generate list of unique values in contributing factor columns
# unique_cf_values = df_data_100000["contributing_factor_vehicle_1"].unique().tolist()

# # create categorical dtypes out of the categories and assign cf category to column:
# list_columns_to_categorise = columns_to_be_unspecified
# create_categories(list_columns_to_categorise)


# print(df_data_100000)

df_data_100000 = pd.get_dummies(df_data_100000, columns=["contributing_factor_vehicle_1"])

# Matts code:

def narrowing_down_factor(factor):
    distraction = ['Driver Inattention/Distraction','Passenger Distraction','Fell Asleep','Outside Car Distraction',
                   'Fatigued/Drowsy','Cell Phone (hand-Held)','Using On Board Navigation Device',
                   'Tinted Windows','Eating or Drinking', 'Other Electronic Device', 'Cell Phone (hands-free)',
                   'Listening/Using Headphones']
    driver_mistake = ['Following Too Closely','Failure to Yield Right-of-Way','Backing Unsafely',
                      'Passing or Lane Usage Improper','Passing Too Closely','Unsafe Lane Changing',
                      'Turning Improperly','Driver Inexperience', 'Failure to Keep Right',
                      'Driverless/Runaway Vehicle', 'Oversized Vehicle']
    illegal_action = ['Unsafe Speed','Alcohol Involvement','Traffic Control Disregarded','Aggressive Driving/Road Rage',
                      'Drugs (illegal)']
    other_involvement =  ['Other Vehicular','Reaction to Uninvolved Vehicle', 'Animals Action',
                          'View Obstructed/Limited','Pedestrian/Bicyclist/Other Pedestrian Error/Confusion',
                           'Glare','Vehicle Vandalism']
    bad_road_infrastructure = ['Pavement Slippery', 'Obstruction/Debris','Pavement Defective',
                               'Other Lighting Defects']
    car_failures = ['Brakes Defective', 'Steering Failure', 'Tire Failure/Inadequate',
                    'Traffic Control Device Improper/Non-Working', 'Lane Marking Improper/Inadequate',
                    'Tow Hitch Defective','Headlights Defective', 'Shoulders Defective/Improper',
                    'Windshield Inadequate']
    medical = ['Lost Consciousness', 'Illnes','Accelerator Defective','Physical Disability','Prescription Medication']
    if factor in medical:
        return "medical"
    elif factor in car_failures:
        return "car_failure"
    elif factor in bad_road_infrastructure:
        return "bad_road_infrastructure"
    elif factor in other_involvement:
        return "other_involvement"
    elif factor in illegal_action:
        return "illegal_action"
    elif factor in driver_mistake:
        return "driver_mistake"
    elif factor in distraction:
        return "distraction"

# To apply fuction on dataset to get combine factors
df_data_100000['contributing_factor_1'] = df_data_100000['contributing_factor_vehicle_1'].apply(lambda x: narrowing_down_factor(x))
df_data_100000['contributing_factor_2'] = df_data_100000['contributing_factor_vehicle_2'].apply(lambda x: narrowing_down_factor(x))
df_data_100000['contributing_factor_3'] = df_data_100000['contributing_factor_vehicle_3'].apply(lambda x: narrowing_down_factor(x))
df_data_100000['contributing_factor_4'] = df_data_100000['contributing_factor_vehicle_4'].apply(lambda x: narrowing_down_factor(x))
df_data_100000['contributing_factor_5'] = df_data_100000['contributing_factor_vehicle_5'].apply(lambda x: narrowing_down_factor(x))
This code apply after i used Matthew's code
# To apply one hot encoding
contribute_factor_1 = pd.get_dummies(df_data_100000.contributing_factor_1, prefix='contributing_factor_1', drop_first="True")
contribute_factor_2 = pd.get_dummies(df_data_100000.contributing_factor_2, prefix='contributing_factor_2', drop_first="True")
contribute_factor_3 = pd.get_dummies(df_data_100000.contributing_factor_3, prefix='contributing_factor_3', drop_first="True")
contribute_factor_4 = pd.get_dummies(df_data_100000.contributing_factor_4, prefix='contributing_factor_4', drop_first="True")
contribute_factor_5 = pd.get_dummies(df_data_100000.contributing_factor_5, prefix='contributing_factor_5', drop_first="True")
# To make a dataframe individual
df1 = pd.concat([contribute_factor_1], axis=1)
df2 = pd.concat([contribute_factor_2], axis=1)
df3 = pd.concat([contribute_factor_3], axis=1)
df4 = pd.concat([contribute_factor_4], axis=1)
df5 = pd.concat([contribute_factor_5], axis=1)

# To merge all dataframe in the original dataframe
result = pd.concat([df_data_100000, df1, df2, df3, df4, df5], axis=1)

# regrouping similar columns in the vehicle section:
# df_data_100000 = pd.get_dummies(df_data_100000, columns=["vehicle_type_code1", "vehicle_type_code2", "vehicle_type_code_3", "vehicle_type_code_4", "vehicle_type_code_5"])


# medical = df_data_100000.groupby(['Lost Consciousness', 'Illnes', 'Accelerator Defective', 'Physical Disability', 'Prescription Medication'])

# check the dtype
print(df_data_100000.describe())

# lowercase
columns_to_be_lowercased = columns_to_be_filled
df_data_100000[columns_to_be_lowercased] = df_data_100000[columns_to_be_lowercased].apply(lambda value: value.astype(str).str.lower())

# make clean csv file
write_to_csv("clean_crashes.csv")

# columns_to_be_lowercased = []
# df_data_100000[columns_to_be_lowercased] = df_data_100000[columns_to_be_lowercased].apply(lambda value: value.astype(str).str.lower())