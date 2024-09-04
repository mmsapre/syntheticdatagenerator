from flask import Flask, request, jsonify, Response
from sdv.tabular import CopulaGAN, TVAESynthesizer, CTGAN, GaussianCopulaSynthesizer, FastMLESynthesizer
import pandas as pd
import pickle
import os
import io
import pandas as pd
from cryptography.fernet import Fernet
import psycopg2
import xml.etree.ElementTree as ET
import io

app = Flask(__name__)

MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

key = Fernet.generate_key()
cipher_suite = Fernet(key)

templates = {}
secure_templates = {}
users = {}
user_roles = {"admin": "all_access", "user": "restricted_access"}

conn_params = {
    'dbname': 'your_db_name',
    'user': 'your_db_user',
    'password': 'your_db_password',
    'host': 'localhost',
    'port': '5432'
}

def fetch_data_dictionary():
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    cursor.execute("SELECT field_name, description, restricted FROM data_dictionary")
    rows = cursor.fetchall()

    data_dictionary = {}
    for row in rows:
        field_name, description, restricted = row
        data_dictionary[field_name] = {
            "description": description,
            "restricted": restricted
        }

    cursor.close()
    conn.close()

    return data_dictionary


def parse_xml(xml_data):
    root = ET.fromstring(xml_data)
    data = {}
    for child in root:
        data[child.tag] = [elem.text for elem in child]
    return data


@app.route('/upload-data-dictionary', methods=['POST'])
def upload_data_dictionary():
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_dictionary (
            field_name VARCHAR(255) PRIMARY KEY,
            description TEXT,
            restricted BOOLEAN
        )
    """)

    data_dict = request.json.get('data_dictionary', {})

    for field_name, details in data_dict.items():
        cursor.execute("""
            INSERT INTO data_dictionary (field_name, description, restricted)
            VALUES (%s, %s, %s)
            ON CONFLICT (field_name) DO UPDATE
            SET description = EXCLUDED.description,
                restricted = EXCLUDED.restricted
        """, (field_name, details["description"], details["restricted"]))

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Data dictionary uploaded to PostgreSQL"}), 201


def get_model(model_name, **kwargs):
    if model_name == 'CopulaGAN':
        return CopulaGAN(**kwargs)
    elif model_name == 'TVAESynthesizer':
        return TVAESynthesizer(**kwargs)
    elif model_name == 'CTGAN':
        return CTGAN(**kwargs)
    elif model_name == 'GaussianCopulaSynthesizer':
        return GaussianCopulaSynthesizer(**kwargs)
    elif model_name == 'FastMLESynthesizer':
        return FastMLESynthesizer(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@app.route('/train-model', methods=['POST'])
def train_model():
    model_name = request.json.get('model_name', 'CopulaGAN')
    model_params = request.json.get('model_params', {})
    training_data = request.json.get('training_data')

    if not training_data:
        return jsonify({"error": "Training data is required"}), 400

    df = pd.DataFrame(training_data)

    model = get_model(model_name, **model_params)

    model.fit(df)

    model_filename = f"{model_name}.pkl"
    with open(os.path.join(MODEL_DIR, model_filename), 'wb') as model_file:
        pickle.dump(model, model_file)

    return jsonify({"message": f"Model {model_name} trained and saved as {model_filename}"}), 200


@app.route('/generate-data-using-model', methods=['POST'])
def generate_data_using_model():
    model_name = request.json.get('model_name', 'CopulaGAN')
    num_rows = int(request.json.get('num_rows', 100))

    model_filename = f"{model_name}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model {model_name} not found. Please train the model first."}), 404

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    synthetic_data = model.sample(num_rows)

    accuracy = model.evaluate(synthetic_data)

    return jsonify({
        "synthetic_data": synthetic_data.to_dict(),
        "accuracy": accuracy
    }), 200


def generate_jmeter_script(synthetic_data, output_format):
    jmeter_template = f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <jmeterTestPlan version="1.2" properties="5.0" jmeter="5.4.1">
        <hashTree>
            <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="Test Plan" enabled="true">
                <stringProp name="TestPlan.comments"></stringProp>
                <boolProp name="TestPlan.functional_mode">false</boolProp>
                <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
                <elementProp name="TestPlan.user_defined_variables" elementType="Arguments" guiclass="ArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
                    <collectionProp name="Arguments.arguments"/>
                </elementProp>
                <stringProp name="TestPlan.user_define_classpath"></stringProp>
            </TestPlan>
            <hashTree>
                <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Thread Group" enabled="true">
                    <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
                    <elementProp name="ThreadGroup.num_threads" elementType="Arguments">
                        <intProp name="ThreadGroup.num_threads">1</intProp>
                    </elementProp>
                    <longProp name="ThreadGroup.ramp_time">1</longProp>
                    <boolProp name="ThreadGroup.scheduler">false</boolProp>
                    <stringProp name="ThreadGroup.duration"></stringProp>
                    <stringProp name="ThreadGroup.delay"></stringProp>
                </ThreadGroup>
                <hashTree>
                    <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="HTTP Request" enabled="true">
                        <elementProp name="HTTPSamplerProxy.arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" enabled="true">
                            <collectionProp name="Arguments.arguments"/>
                        </elementProp>
                        <stringProp name="HTTPSampler.domain"></stringProp>
                        <stringProp name="HTTPSampler.port"></stringProp>
                        <stringProp name="HTTPSampler.protocol"></stringProp>
                        <stringProp name="HTTPSampler.path">/api/test</stringProp>
                        <stringProp name="HTTPSampler.method">POST</stringProp>
                        <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
                        <boolProp name="HTTPSampler.auto_redirects">false</boolProp>
                        <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
                        <boolProp name="HTTPSampler.DO_MULTIPART_POST">false</boolProp>
                        <stringProp name="HTTPSampler.embedded_url_re"></stringProp>
                    </HTTPSamplerProxy>
                    <hashTree/>
                </hashTree>
            </hashTree>
        </hashTree>
    </jmeterTestPlan>
    """
    return jmeter_template


@app.route('/generate-jmeter-script', methods=['POST'])
def generate_jmeter_script_endpoint():
    synthetic_data = request.json.get('synthetic_data', {})
    output_format = request.json.get('format', 'json')  # Default to JSON

    jmeter_script = generate_jmeter_script(synthetic_data, output_format)

    return Response(jmeter_script, mimetype='application/xml',
                    headers={"Content-Disposition": "attachment;filename=jmeter_test_plan.jmx"})

@app.route('/upload-data-dictionary', methods=['POST'])
def upload_data_dictionary():
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_dictionary (
            field_name VARCHAR(255) PRIMARY KEY,
            description TEXT,
            restricted BOOLEAN
        )
    """)

    data_dict = request.json.get('data_dictionary', {})

    for field_name, details in data_dict.items():
        cursor.execute("""
            INSERT INTO data_dictionary (field_name, description, restricted)
            VALUES (%s, %s, %s)
            ON CONFLICT (field_name) DO UPDATE
            SET description = EXCLUDED.description,
                restricted = EXCLUDED.restricted
        """, (field_name, details["description"], details["restricted"]))

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Data dictionary uploaded to PostgreSQL"}), 201


@app.route('/upload-data-sample', methods=['POST'])
def upload_data_sample():
    content_type = request.headers.get('Content-Type')
    data_dictionary = fetch_data_dictionary()

    if 'application/json' in content_type:
        dataset = request.json
    elif 'text/csv' in content_type:
        dataset = pd.read_csv(io.StringIO(request.data.decode('utf-8'))).to_dict(orient='list')
    elif 'application/xml' in content_type or 'text/xml' in content_type:
        dataset = parse_xml(request.data.decode('utf-8'))
    else:
        return jsonify({"error": "Unsupported content type"}), 400

    # Data quality report
    report = {
        "missing_fields": [],
        "extra_fields": [],
        "type_mismatches": [],
        "restricted_data_issues": []
    }

    # Check for missing and extra fields
    for field in data_dictionary.keys():
        if field not in dataset:
            report["missing_fields"].append(field)

    for field in dataset.keys():
        if field not in data_dictionary:
            report["extra_fields"].append(field)

    # Validate each field in the dataset
    for field, values in dataset.items():
        if field in data_dictionary:
            # Check for restricted data
            if data_dictionary[field]["restricted"]:
                if any(value != "RESTRICTED" for value in values):
                    report["restricted_data_issues"].append(field)

            # Simple type validation check (categorical vs continuous)
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                expected_type = 'categorical'
            elif isinstance(values, list) and all(isinstance(value, (int, float)) for value in values):
                expected_type = 'continuous'
            else:
                expected_type = 'unknown'

            if field in templates:
                for template_field in templates[field]["fields"]:
                    if template_field["type"] == "categorical" and expected_type != "categorical":
                        report["type_mismatches"].append({
                            "field": field,
                            "expected": "categorical",
                            "found": expected_type
                        })
                    elif template_field["type"] == "continuous" and expected_type != "continuous":
                        report["type_mismatches"].append({
                            "field": field,
                            "expected": "continuous",
                            "found": expected_type
                        })

    return jsonify(report), 200


if __name__ == "__main__":
    app.run(debug=True)
