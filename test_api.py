import aiohttp
import asyncio

async def test_pdf_processing():
    async with aiohttp.ClientSession() as session:
        # Process PDF
        data = aiohttp.FormData()
        data.add_field('file',
                      open('policy.pdf', 'rb'),
                      filename='policy.pdf')
        data.add_field('prompt_type', 'policy_json_conversion')
        
        async with session.post('http://localhost:8000/api/process-pdf',
                              data=data) as response:
            result = await response.json()
            task_id = result['task_id']
            
        # Poll for results
        while True:
            async with session.get(f'http://localhost:8000/api/pdf-status/{task_id}') as response:
                status = await response.json()
                if status['status'] == 'completed':
                    print(status['result'])
                    break
                elif status['status'] == 'failed':
                    print(f"Error: {status.get('error')}")
                    break
            await asyncio.sleep(5)

asyncio.run(test_pdf_processing())