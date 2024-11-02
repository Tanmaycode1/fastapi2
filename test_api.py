import aiohttp
import asyncio
import os
from pathlib import Path

async def test_pdf_processing(pdf_path: str = 'policy.pdf', base_url: str = 'http://localhost:8000'):
    """
    Test PDF processing endpoint with proper error handling
    
    Args:
        pdf_path: Path to the PDF file
        base_url: Base URL of the API
    """
    # Verify PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
    try:
        async with aiohttp.ClientSession() as session:
            # Process PDF
            data = aiohttp.FormData()
            try:
                with open(pdf_path, 'rb') as f:
                    data.add_field('file',
                                 f,
                                 filename=Path(pdf_path).name)
                    data.add_field('prompt_type', 'policy_json_conversion')
                
                    async with session.post(f'{base_url}/api/process-pdf',
                                          data=data) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"PDF upload failed with status {response.status}: {error_text}")
                            
                        result = await response.json()
                        task_id = result['task_id']
                        print(f"PDF upload successful. Task ID: {task_id}")
            except aiohttp.ClientError as e:
                raise Exception(f"Error uploading PDF: {str(e)}")
                
            # Poll for results
            max_retries = 120  # 10 minutes maximum waiting time
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    async with session.get(f'{base_url}/api/pdf-status/{task_id}') as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Status check failed with status {response.status}: {error_text}")
                            
                        status = await response.json()
                        
                        if status['status'] == 'completed':
                            print("\nProcessing completed successfully!")
                            if 'text_content' in status:
                                text_content = status['text_content']
                                print(f"\nExtracted text from PDF ({len(text_content)} characters):")
                                print(f"\n{'-'*80}\n")
                                print(text_content)
                                print(f"\n{'-'*80}\n")
                                return text_content
                            else:
                                print("\nWarning: No text content in response")
                                return status
                        elif status['status'] == 'failed':
                            raise Exception(f"Processing failed: {status.get('error', 'Unknown error')}")
                        
                        print(f"Status: {status['status']}. Waiting... (Attempt {retry_count + 1}/{max_retries})")
                        
                except aiohttp.ClientError as e:
                    print(f"Error checking status: {str(e)}")
                    
                retry_count += 1
                await asyncio.sleep(5)
                
            raise TimeoutError("Processing timed out after 10 minutes")
            
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        result = asyncio.run(test_pdf_processing())
        if isinstance(result, str):
            print(f"\nPDF Processing Successful! Extracted {len(result)} characters of text.")
        else:
            print("\nPDF Processing Successful, but received unexpected response format:", result)
    except Exception as e:
        print(f"Test failed: {str(e)}")