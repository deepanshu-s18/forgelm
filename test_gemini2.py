from google import genai

client = genai.Client(api_key="AQ.Ab8RN6IzCNLFOOI1KR6SO1zk2iZmXnK9WHIxvlSjuhu7a_VGeg")
for m in client.models.list():
    if "flash" in m.name:
        print(m.name)
