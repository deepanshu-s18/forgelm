from google import genai

client = genai.Client(api_key="AQ.Ab8RN6IzCNLFOOI1KR6SO1zk2iZmXnK9WHIxvlSjuhu7a_VGeg")

def test(model):
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Say hello"
        )
        print(f"{model}: SUCCESS")
    except Exception as e:
        print(f"{model}: FAILED - {e}")

test("gemini-3.5-flash")
test("gemini-3.5-flash-lite")
test("gemini-3.1-flash-lite")
