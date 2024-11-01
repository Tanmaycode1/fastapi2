# Medical Policy Analysis API

A FastAPI application for medical policy analysis using GPT-4, featuring multiple specialized endpoints for different types of analysis and processing.

## API Endpoints

### 1. Policy Encoding
```http
POST /api/generate-encoding
```
Converts medical policy documents into structured JSON format with criteria and logical relationships.

**Request:**
```json
{
    "body": "Medical Policy Text...",
    "max_tokens": 4000,
    "temperature": 0.7
}
```

**Response:**
```json
{
    "Source": "filename",
    "Name": "policyname",
    "case_scenario": "",
    "encoding_issues": "",
    "criteria": [
        {
            "no": "1",
            "var": "age_requirement",
            "crit": "Patient is 18 years or older"
        }
    ],
    "logic": "AND(OR(age_requirement), NOT(contraindications))"
}
```

### 2. Claim Review
```http
POST /api/generate-claim-review
```
Analyzes medical claims against policy requirements.

**Request:**
```json
{
    "body": [
        "Policy Document Text...",
        "Claim Details..."
    ],
    "max_tokens": 4000,
    "temperature": 0.7
}
```

**Response:**
```json
{
    "claim_comparison": [
        {
            "policy_criterion": "Maximum hospital stay: 7 days",
            "extracted_patient_data": "Hospital Stay: 9 days",
            "matching_status": "Not Met",
            "explanation": "Exceeds by 2 days..."
        }
    ],
    "error_detection": [
        {
            "error_type": "Duplicate Charge",
            "service": "X-Ray",
            "explanation": "Service billed twice..."
        }
    ],
    "approval_status": "Requires Additional Review"
}
```

### 3. Recommendation and Mapping
```http
POST /api/generate-recommendation-and-mapping
```
Generates treatment recommendations based on policy criteria.

**Request:**
```json
{
    "body": {
        "policy": [
            {
                "no": "1",
                "var": "condition_severity",
                "crit": "Severe condition requiring intervention"
            }
        ],
        "patient_data": "Patient history and current condition..."
    }
}
```

**Response:**
```json
{
    "Mapping": [
        {
            "no": "1",
            "var": "condition_severity",
            "supporting_data": "Patient presents with severe symptoms...",
            "evaluation": true,
            "rationale": "Symptoms meet severity criteria"
        }
    ],
    "Final Decision": {
        "recommendation": "Approved",
        "decision_rationale": "Patient meets all criteria...",
        "confidence_score": "95"
    }
}
```

### 4. Alternative Care Pathway
```http
POST /api/generate-alternative-care-pathway
```
Suggests alternative treatment options.

**Request:**
```json
{
    "body": {
        "PatientData": "Patient condition and history...",
        "requestedProcedure": "97140"
    }
}
```

**Response:**
```json
[
    {
        "procedure_code": "97110",
        "alternative": "Therapeutic Exercise",
        "rationale": "Improves flexibility and strength",
        "auto_approval": false
    }
]
```

### 5. Session Management

#### Start Session
```http
POST /api/start-session
```

**Request:**
```json
{
    "body": "Initial context or policy information..."
}
```

**Response:**
```json
{
    "session_id": "uuid-string"
}
```

#### Send Message
```http
POST /api/send-message
```

**Request:**
```json
{
    "session_id": "uuid-string",
    "message": "User query or input..."
}
```

**Response:**
```json
{
    "response": "Assistant's response..."
}
```

### 6. Health Check
```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## Error Responses

All endpoints return error responses in this format:
```json
{
    "error": "Error description",
    "type": "error_type"
}
```

## Features

### Chunked Processing
- Handles large responses through chunking
- Maintains context across chunks
- Merges partial responses intelligently

### Session Management
- Maintains conversation context
- Automatic cleanup of old sessions
- Stateful interactions

### Error Handling
- Retry mechanism with exponential backoff
- JSON validation and repair
- Detailed error logging

### Processing Features
- Context preservation
- Intelligent chunking
- Response validation
- Format verification

## Technical Notes

1. **Response Processing**:
   - Maximum 5 chunks per response
   - Automatic JSON structure repair
   - Duplicate detection in merged responses

2. **Session Handling**:
   - 24-hour session timeout
   - Hourly cleanup of expired sessions
   - Context maintenance across requests

3. **Error Recovery**:
   - 3 retry attempts with exponential backoff
   - Partial response recovery
   - Context preservation on errors

4. **Performance**:
   - 30-second timeout per request
   - Chunked processing for large responses
   - Efficient memory management

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [project-directory]
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python -m uvicorn app.main:app --reload
```

## Testing

### Using curl

1. Test Policy Encoding:
```bash
curl -X POST http://localhost:8000/api/generate-encoding \
  -H "Content-Type: application/json" \
  -d '{
    "body": "Medical Policy Content...",
    "max_tokens": 4000,
    "temperature": 0.7
  }'
```

2. Test Claim Review:
```bash
curl -X POST http://localhost:8000/api/generate-claim-review \
  -H "Content-Type: application/json" \
  -d '{
    "body": ["Policy Document", "Claim Details"],
    "max_tokens": 4000
  }'
```

### Using Python

```python
import requests

def test_encoding():
    response = requests.post(
        "http://localhost:8000/api/generate-encoding",
        json={
            "body": "Medical Policy Content...",
            "max_tokens": 4000
        }
    )
    print(response.json())

def test_session():
    # Start session
    session_response = requests.post(
        "http://localhost:8000/api/start-session",
        json={"body": "Initial context..."}
    )
    session_id = session_response.json()["session_id"]
    
    # Send message
    message_response = requests.post(
        "http://localhost:8000/api/send-message",
        json={
            "session_id": session_id,
            "message": "Query..."
        }
    )
    print(message_response.json())
```

## API Rate Limits and Performance

### Limits
- Maximum request size: 8000 tokens
- Maximum response chunks: 5
- Session timeout: 24 hours
- Request timeout: 30 seconds

### Performance Optimization
1. **Chunk Size Management**:
   - Default: 4000 tokens
   - Adjustable via max_tokens parameter
   - Automatic chunking for large responses

2. **Session Optimization**:
   - Automatic context pruning
   - Memory efficient storage
   - Regular cleanup of inactive sessions

3. **Error Recovery**:
   - Exponential backoff retry
   - Partial response recovery
   - Context preservation

## Best Practices

### Making Requests
1. **Policy Encoding**:
   - Keep policy text clear and structured
   - Include all relevant sections
   - Specify clear criteria boundaries

2. **Claim Review**:
   - Provide complete policy documents
   - Include detailed claim information
   - Specify all relevant codes

3. **Session Management**:
   - Maintain active sessions
   - Clear unused sessions
   - Handle timeouts gracefully

### Error Handling
1. **Client-Side**:
   - Implement retry logic
   - Handle timeout errors
   - Validate responses

2. **Response Processing**:
   - Validate JSON structure
   - Handle partial responses
   - Merge multi-part responses

## Response Formats

### Success Responses
All successful responses will have HTTP status code 200 and contain:
- Valid JSON data
- Complete response structure
- Required fields based on endpoint

### Error Responses
Error responses include:
- HTTP status code (4xx or 5xx)
- Error type identifier
- Detailed error message
- Traceable error ID

## Security Considerations

1. **API Key Management**:
   - Store securely in .env
   - Rotate regularly
   - Never expose in code

2. **Data Protection**:
   - No PII storage
   - Session data encryption
   - Regular session cleanup

3. **Access Control**:
   - Rate limiting
   - Session validation
   - Request validation

## Troubleshooting

### Common Issues

1. **Connection Errors**:
```bash
# Check API status
curl http://localhost:8000/health
```

2. **Token Errors**:
- Verify API key in .env
- Check token limits
- Monitor usage

3. **Response Errors**:
- Check request format
- Validate input data
- Review error messages

### Debug Mode
Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```
