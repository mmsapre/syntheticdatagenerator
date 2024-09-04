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
    template_name = request.json.get('template_name')
    num_rows = int(request.json.get('num_rows', 100))
    epochs = int(request.json.get('epochs', 300))
    batch_size = int(request.json.get('batch_size', 32))
    generator_lr = float(request.json.get('generator_lr', 2e-4))
    discriminator_lr = float(request.json.get('discriminator_lr', 2e-4))

    # Limit the number of rows to 500
    if num_rows > 500:
        num_rows = 500

    # Fetch the template
    template = templates.get(template_name)
    if not template:
        return jsonify({"error": "Template not found"}), 404

    # Fetch the data dictionary
    data_dictionary = fetch_data_dictionary()

    # Prepare the dataset according to the template and dictionary
    sample_data = {}
    for field in template["fields"]:
        field_name = field["name"]
        if field_name in data_dictionary:
            if data_dictionary[field_name]["restricted"]:
                sample_data[field_name] = ["RESTRICTED"] * num_rows
            else:
                # Generate dummy data according to field type (this is just an example)
                if field["type"] == "categorical":
                    sample_data[field_name] = [field["categories"][i % len(field["categories"])] for i in
                                               range(num_rows)]
                elif field["type"] == "continuous":
                    min_val, max_val = field["range"]
                    sample_data[field_name] = [min_val + i * (max_val - min_val) / num_rows for i in range(num_rows)]

    # Convert to DataFrame
    df = pd.DataFrame(sample_data)

    # Define the GAN model
    model = CopulaGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr
    )

    # Train the GAN model on the prepared dataset
    model.fit(df)

    # Generate synthetic data
    synthetic_data = model.sample(num_rows)

    # Evaluate the accuracy of the generated data
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


@app.route('/validate-dataset', methods=['POST'])
def validate_dataset():
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

            # Type validation (example: categorical vs continuous)
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                expected_type = 'categorical'
            elif isinstance(values, list) and all(isinstance(value, (int, float)) for value in values):
                expected_type = 'continuous'
            else:
                expected_type = 'unknown'

            # Simple type validation check
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


# API to add users to roles
@app.route('/add-user', methods=['POST'])
def add_user():
    username = request.json.get('username')
    role = request.json.get('role')

    if role not in user_roles:
        return jsonify({"error": "Invalid role"}), 400

    users[username] = role
    return jsonify({"message": f"User {username} added with role {role}"}), 201


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

@app.route('/template', methods=['POST'])
def create_template():
    template_name = request.json.get('template_name')
    fields = request.json.get('fields')

    if template_name in templates:
        return jsonify({"error": "Template already exists"}), 400

    # Assign sample data from the dictionary without handling restricted data
    for field in fields:
        if field["name"] in data_dictionary:
            field["description"] = data_dictionary[field["name"]]["description"]
        else:
            return jsonify({"error": f"Field {field['name']} not found in data dictionary"}), 400

    templates[template_name] = {"fields": fields}
    return jsonify({"message": "Template created successfully"}), 201

# API to generate data using a general template
@app.route('/generate-template/<template_name>', methods=['POST'])
def generate_data_from_template(template_name):
    if template_name not in templates:
        return jsonify({"error": "Template not found"}), 404

    template = templates[template_name]
    num_rows = request.json.get('num_rows', 10)

    data = {}
    for field in template["fields"]:
        if field["type"] == "categorical":
            data[field["name"]] = pd.Series([field["categories"][i % len(field["categories"])] for i in range(num_rows)])
        elif field["type"] == "continuous":
            min_val, max_val = field["range"]
            data[field["name"]] = pd.Series([min_val + i * (max_val - min_val) / num_rows for i in range(num_rows)])

    generated_data = pd.DataFrame(data)
    return jsonify(generated_data.to_dict()), 200

# API to create a new secure template with RBAC and restricted data handling
@app.route('/secure-template', methods=['POST'])
def create_secure_template():
    template_name = request.json.get('template_name')
    fields = request.json.get('fields')
    user_role = request.headers.get('Role', 'user')  # Assuming role is passed in the header

    if template_name in secure_templates:
        return jsonify({"error": "Template already exists"}), 400

    # Assign sample data from the dictionary and identify restricted data
    for field in fields:
        if field["name"] in data_dictionary:
            if not check_access(user_role, field["name"]):
                return jsonify({"error": f"Access denied for field: {field['name']}"}), 403

            field["description"] = data_dictionary[field["name"]]["description"]
            field["restricted"] = data_dictionary[field["name"]]["restricted"]
        else:
            return jsonify({"error": f"Field {field['name']} not found in data dictionary"}), 400

    secure_templates[template_name] = {"fields": fields}
    return jsonify({"message": "Secure template created successfully with sample data"}), 201

# API to generate data using a secure template with restricted data handling
@app.route('/generate-secure-template/<template_name>', methods=['POST'])
def generate_data_from_secure_template(template_name):
    if template_name not in secure_templates:
        return jsonify({"error": "Secure template not found"}), 404

    template = secure_templates[template_name]
    num_rows = request.json.get('num_rows', 10)
    user_role = request.headers.get('Role', 'user')  # Assuming role is passed in the header

    data = {}
    for field in template["fields"]:
        if not check_access(user_role, field["name"]):
            data[field["name"]] = pd.Series(["RESTRICTED" for _ in range(num_rows)])
        elif field["restricted"]:
            # Handle restricted data: e.g., anonymize or apply differential privacy
            data[field["name"]] = pd.Series([cipher_suite.encrypt(b"RESTRICTED").decode("utf-8") for _ in range(num_rows)])
        elif field["type"] == "categorical":
            data[field["name"]] = pd.Series([field["categories"][i % len(field["categories"])] for i in range(num_rows)])
        elif field["type"] == "continuous":
            min_val, max_val = field["range"]
            data[field["name"]] = pd.Series([min_val + i * (max_val - min_val) / num_rows for i in range(num_rows)])

    generated_data = pd.DataFrame(data)
    return jsonify(generated_data.to_dict()), 200

if __name__ == "__main__":
    app.run(debug=True)
