import requests
import json


def test_policy_parsing():
    url = "http://localhost:8000/api/convert-policy-to-json"

    payload = {
        "body": """ixekizumab (Taltz®)
Policy # 00513
Original Effective Date: 09/01/2016
Current Effective Date: 02/14/2022

Based on review of available data, the Company may consider ixekizumab (Taltz®)‡ for the treatment of patients with plaque psoriasis to be eligible for coverage.

Patient Selection Criteria
Coverage eligibility for ixekizumab (Taltz) will be considered when the following criteria are met:
• Patient has a diagnosis of moderate to severe plaque psoriasis; AND
• Patient is 6 years of age or older; AND
• Patient has a negative TB (tuberculosis) test (e.g., purified protein derivative [PPD], blood test) prior to treatment; AND
• Patient is a candidate for phototherapy or systemic therapy; AND
• Requested drug is NOT used in combination with other biologic disease-modifying anti-rheumatic drugs (DMARDs), such as adalimumab (Humira®)‡ or etanercept (Enbrel®)‡ OR other drugs such as apremilast (Otezla®)‡ or tofacitinib (Xeljanz®/XR)‡; AND
• Patient has greater than 10% of body surface area (BSA) OR less than or equal to 10% BSA with plaque psoriasis involving sensitive areas or areas that would significantly impact daily function (such as palms, soles of feet, head/neck or genitalia); AND
• For patients 18 years of age or older: Patient has failed treatment with TWO of the following after at least TWO months of therapy with EACH product: adalimumab (Humira), etanercept (Enbrel), apremilast (Otezla), ustekinumab (Stelara), secukinumab (Cosentyx), guselkumab (Tremfya), or risankizumab (Skyrizi); AND
• Patient has failed to respond to an adequate trial of one of the following treatment modalities:
    o Ultraviolet B; or
    o Psoralen positive Ultraviolet A; or
    o Systemic therapy (e.g., methotrexate, cyclosporine, acitretin)""",
        "max_tokens": 4000,
        "temperature": 0.7
    }

    response = requests.post(url, json=payload)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    test_policy_parsing()