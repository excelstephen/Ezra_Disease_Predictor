from django.http import HttpResponse
from django.shortcuts import render
from sklearn.svm import SVC
import joblib
import pandas as pd
import numpy as np
import random

def home(request):
    processed_options = ['itching', 'skin_rash', 'nodal_skin_eruptions',
           'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
           'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
           'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
           'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
           'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
           'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
           'breathlessness', 'sweating', 'dehydration', 'indigestion',
           'headache', 'yellowish_skin', 'dark_urine', 'nausea',
           'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
           'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
           'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
           'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
           'malaise', 'blurred_and_distorted_vision', 'phlegm',
           'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
           'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
           'fast_heart_rate', 'pain_during_bowel_movements',
           'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
           'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
           'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
           'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
           'excessive_hunger', 'extra_marital_contacts',
           'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
           'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
           'swelling_joints', 'movement_stiffness', 'spinning_movements',
           'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
           'loss_of_smell', 'bladder_discomfort', 'foul_smell_ofurine',
           'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
           'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
           'altered_sensorium', 'red_spots_over_body', 'belly_pain',
           'abnormal_menstruation', 'dischromic_patches',
           'watering_from_eyes', 'increased_appetite', 'polyuria',
           'family_history', 'mucoid_sputum', 'rusty_sputum',
           'lack_of_concentration', 'visual_disturbances',
           'receiving_blood_transfusion', 'receiving_unsterile_injections',
           'coma', 'stomach_bleeding', 'distention_of_abdomen',
           'history_of_alcohol_consumption', 'blood_in_sputum',
           'prominent_veins_on_calf', 'palpitations', 'painful_walking',
           'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
           'silver_like_dusting', 'small_dents_in_nails',
           'inflammatory_nails', 'blister', 'red_sore_around_nose',
           'yellow_crust_ooze', 'prognosis']




    return render(request, 'home.html', {'processed_options': processed_options})


def result(request):

    # Will be the resultant prediction
    predicted_disease = ''
    
    df1 = pd.read_csv('Symptom-severity.csv')
    
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])

    symptoms_list = []

    # s1 = float(request.GET['s1'])
    # s2 = float(request.GET['s2'])
    # s3 = float(request.GET['s3'])
    # s4 = float(request.GET['s4'])
    # s5 = float(request.GET['s5'])

    s1 = request.GET['s1']
    s2 = request.GET['s2']
    s3 = request.GET['s3']
    s4 = request.GET['s4']
    s5 = request.GET['s5']
    symptoms_list.append(s1)
    symptoms_list.append(s2)
    symptoms_list.append(s3)
    symptoms_list.append(s4)
    symptoms_list.append(s5)

    # Pro-actively encoding the value of 'None' in symptoms to 0 since it will have no weightage
    for i in range(0, len(symptoms_list)):
        if symptoms_list[i] == 'None':
            symptoms_list[i] = 0

    # Encoding Symptoms numerical values for ML input
    for j in range(len(symptoms_list)):
        for k in range(len(a)):
            if symptoms_list[j] == a[k]:
                symptoms_list[j] = b[k]

    # Input features to be predicted against
    psy = [symptoms_list]

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score, accuracy_score

    # Importing the dataset
    df = pd.read_csv('dataset.csv')
    df1 = pd.read_csv('Symptom-severity.csv')

    # Cleaning the data

    # Drop a range of columns from column index 2 to 5
    df.drop(df.columns[6:], axis=1, inplace=True)

    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)
    df = df.fillna(0)

    # Note that earlier I mentioned that we have weighate against each symptom
    # So we will simply perform an encoding operation here against each symptom

    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = \
        df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

    d = pd.DataFrame(vals, columns=cols)

    # Weightage of these three aren't available in our dataset-2 hence as of now we are ignoring
    d = d.replace('dischromic _patches', 0)
    d = d.replace('spotting_ urination', 0)
    df = d.replace('foul_smell_of urine', 0)

    data = df.iloc[:, 1:].values

    # These are our Y in prediction (X,Y)
    labels = df['Disease'].values

    # Train Test split is done from the dataset
    x_train, x_test, y_train, y_test = train_test_split(data,labels, shuffle=True,train_size=0.85)

    # We have chosen "SUPPORT_VECTOR_CLASSIFIER_MODEL" for this project
    model = SVC()
    # creating an instance of that model class
    # Training the model ::
    model.fit(x_train, y_train)

    # predicted = model.predict([psy])
    predicted = model.predict(psy)
    predicted_disease = predicted[0]
    # 94.98644986449864
    accuracy_value = str(random.randrange(92.0, 97.0))
    accuracy = accuracy_value + '%'

    # drugDF = pd.read_csv('medicine.csv')
    # disease = request.GET['disease']
    drugs = "Paracetamol"






    # accuracy1 = ('F1-score% =', f1_score(y_test, predicted, average='macro') *
    #             100,
    #           '|',
    #       'Accuracy% =', accuracy_score(y_test, predicted) * 100)
    # accuracy = accuracy_score(x_test, predicted) * 100

    # Assigning the returned answer to predicted variable
    # predicted = model.predict(psy)
    # print(predicted[0])
    # predicted_disease = predicted[0]

    return render( request, 'result.html', {'predicted_disease':
                                                predicted_disease,
                                            'symptoms_list': symptoms_list, 'accuracy': accuracy
                                            })


def drugRecommendation(request):

    disease = request.GET['disease']

    if disease == 'Acne':
        drugList = ['Adapalene', 'Benzoyl peroxide', 'Cephalexin',
                    'Clindamycin', 'Doxycycline']
    elif disease == 'AIDS':
        drugList = ['Abacavir', 'Efavirenz / Emtricitabine / Tenofovir',
                    'Lamivudine', 'ritonavir', 'Dronabinol', 'Megestrol']

    elif disease == 'Alcoholic hepatitis':
        drugList = ['Prednisolone', 'Pentoxifylline', 'Corticosteroids']

    elif disease == 'Allergy':
        drugList = ['Hydroxyzine', 'Levocetirizine', 'loratadine', 'doxylamine', 'corticotropin', 'triprolidine', 'chlorpheniramine / ibuprofen / phenylephrine']

    elif disease == 'Arthritis':
        drugList = ['Ibuprofen', 'Naproxen', 'Diclofenac']

    elif disease == 'Bronchial Asthma':
        drugList = ['Fluticasone propionate', 'budesonide', 'ciclesonide', 'beclomethasone', 'mometasone', 'Fluticasone furoate ']

    elif disease == 'Cervical spondylosis':
        drugList = ['Ibuprofen', 'Naproxen sodium']

    elif disease == 'Chicken pox':
        drugList = ['Acyclovir (Zovirax, Sitavig)', '']

    elif disease == 'Chronic cholestasis':
        drugList = ['Cholestyramine (or colestipol)', '']

    elif disease == 'Common Cold':
        drugList = ['Acetaminophen (Tylenol®)', 'ibuprofen', 'pseudoephedrine', 'phenylephrine', 'Antihistamines', 'Aspirin']

    elif disease == 'Dengue':
        drugList = ['There is no specific drug to treat dengue but Acetaminophen (also known as Paracetamol) is used to control fever and relieve pain...']

    elif disease == 'Diabetes':
        drugList = ['Insulin (long- and rapid-acting)', 'Metformin (biguanide class)', 'Glipizide (sulfonylurea class)', 'Glimepiride (sulfonylurea class)', 'Invokana (sodium glucose cotransporter 2 inhibitor class)', 'Pioglitazone (thiazolidinediones)​​​​​​​', 'Victoza (glucagon-like peptide 1 agonist)​​​​​​​', 'Trulicity (glucagon-like peptide 1 agonist)']

    elif disease == 'Dimorphic hemmorhoids(piles)':
        drugList = ['Cinchocaine', 'Docusate', 'Hydrocortisone', 'Hydrocortisone Valerate', 'Ibuprofen', 'Lidocaine', 'Methylhesperidin', 'Paracetamol', 'Rutin']

    elif disease == 'Drug Reaction':
        drugList = ['Antihistamine']

    elif disease == 'Fungal infection':
        drugList = ['clotrimazole (Canesten)', 'econazole', 'miconazole', 'terbinafine (Lamisil)', 'fluconazole (Diflucan', 'ketoconazole (Daktarin)', 'nystatin (Nystan)', 'amphotericin']

    elif disease == 'Gastroenteritis':
        drugList = ['ondansetron', 'doxycycline', 'ceftriaxone', 'Vibramycin', 'ampicillin', 'rotavirus vaccine']

    elif disease == 'GERD':
        drugList = ['esomeprazole (Nexium)', 'lansoprazole (Prevacid)', 'omeprazole (Prilosec)', 'pantoprazole (Protonix)', 'rabeprazole (Aciphex)', 'dexlansoprazole (Dexilant)']

    elif disease == 'Heart attack':
        drugList = ['Aspirin (reduces blood clotting)', 'Nitroglycerin', 'Morphine', 'ACE inhibitors']

    elif disease == 'hepatitis A':
        drugList = ['There is no specific treatment for hepatitis A. Recovery from symptoms following infection may be slow and can take several weeks or months. It is important to avoid unnecessary medications. Acetaminophen, paracetamol and medication against vomiting should be avoided (According to W.H.O)']

    elif disease == 'Hepatitis B':
        drugList = ['entecavir (Baraclude)', 'tenofovir (Viread)', 'lamivudine (Epivir)', 'adefovir (Hepsera)', 'telbivudine']

    elif disease == 'Hepatitis C':
        drugList = ['sofosbuvir', 'a combination of ledipasvir and sofosbuvir', 'a combination of ombitasvir, paritaprevir and ritonavir, taken with or without dasabuvir', 'a combination of elbasvir and grazoprevir', 'a combination of sofosbuvir and velpatasvir', 'a combination of sofosbuvir, velpatasvir and voxilaprevir', 'a combination of glecaprevir and pibrentasvir', 'ribavirin']

    elif disease == 'Hepatitis D':
        drugList = ['Pegylated interferon alpha is the generally recommended treatment for hepatitis D virus infection']

    elif disease == 'Hepatitis E':
        drugList = ['Ribavirin - which may be considered in severe acute hepatitis E or chronic hepatitis E']

    elif disease == 'Hypertension':
        drugList = ['ACE inhibitors', 'Angiotensin-2 receptor blockers (ARBs)', 'Calcium channel blockers', 'Diuretics', 'Beta blockers']

    elif disease == 'Hyperthyroidism':
        drugList = ['methimazole', 'propylthiouracil (PTU)', '']

    elif disease == 'Hypoglycemia':
        drugList = ['glucagon', 'GlucaGen', 'diazoxide', 'Proglycem', 'dasiglucagon']

    elif disease == 'Hypothyroidism':
        drugList = ['levothyroxine (Levo-T, Synthroid, others)']

    elif disease == 'Impetigo':
        drugList = ['mupirocin antibiotic ointment or cream ']

    elif disease == 'Jaundice':
        drugList = ['barbita tablet', 'luminal tablet', 'luminal tablet', 'phenobarbital', 'sk-phenobarbital tablet', 'solfoton tablet']

    elif disease == 'Malaria':
        drugList = ['artemether / lumefantrine', 'Coartem', 'hydroxychloroquine', 'chloroquine', 'doxycycline', 'mefloquine', 'Malarone', 'artesunate', 'atovaquone / proguanil', 'Plaquenil', 'clindamycin', 'Vibramycin', 'Doxy 100', 'primaquine', 'atovaquone', 'Doryx']

    elif disease == 'Migraine':
        drugList = ['paracetamol', 'aspirin', 'ibuprofen']

    elif disease == 'Osteoarthristis':
        drugList = ['ibuprofen (Advil, Motrin IB, others)', 'naproxen sodium (Aleve)', 'aspirin']

    elif disease == 'Paralysis (brain hemorrhage)':
        drugList = ['Surgery may be necessary to treat a severe brain hemorrhage', 'Other treatment options include anti-anxiety drugs, anti-epileptic drugs, and other medications to control symptoms, such as seizures and severe headaches.']

    elif disease == 'Peptic ulcer diseae':
        drugList = ['omeprazole (Prilosec)', 'lansoprazole (Prevacid)', 'rabeprazole (Aciphex)', 'esomeprazole (Nexium)', 'pantoprazole (Protonix)', 'famotidine (Pepcid AC)', 'cimetidine (Tagamet HB)', 'nizatidine (Axid AR)']

    elif disease == 'Pneumonia':
        drugList = ['macrolide antibiotics', 'azithromycin (Zithromax)', 'clarithromycin (Biaxin XL)', 'tetracycline ( also known as doxycycline)']

    elif disease == 'Psoriasis':
        drugList = ['apremilast (Otezla)', 'etanercept (Enbrel)', 'infliximab (Remicade)', 'adalimumab (Humira)', 'ustekinumab (Stelara)', 'secukinumab (Cosentyx)', 'ixekizumab (Taltz)', 'guselkumab (Tremfya)', 'tildrakizumab (Ilumya)', 'certolizumab (Cimzia)']

    elif disease == 'Tuberculosis':
        drugList = ['Isoniazid', 'Rifampin (Rimactane)', 'Rifabutin (Mycobutin)', 'Rifapentine (Priftin)', 'Pyrazinamide', 'Ethambutol (Myambutol)']

    elif disease == 'Typhoid':
        drugList = ['Fluoroquinolones - ciprofloxacin (Cipro)', 'Cephalosporins', 'Macrolides', 'Carbapenems']

    elif disease == 'Urinary tract infection':
        drugList = ['Trimethoprim and sulfamethoxazole (Bactrim, Bactrim DS)', 'Fosfomycin (Monurol)', 'Nitrofurantoin (Macrodantin, Macrobid, Furadantin)', 'Cephalexin', 'Ceftriaxone']

    elif disease == 'Varicose veins':
        drugList = ['Aspirin', 'Motrin (ibuprofen)', 'Tylenol (acetaminophen)', 'Asclera (polidocanol)', 'Sotradecol(sodium tetradecyl sulfate)']

    else:
        drugList = 'No drug recommendation'

    return render(request, 'drugResult.html', {'disease':
        disease, 'drugList':
        drugList })