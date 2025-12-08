import requests

# results = DDGS().text("python programming", max_results=5)
# for x in results:
#    print(json.dumps(x, indent=4))

r = requests.get(
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    headers={
        "User-Agent": "User-Agent: CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0"
    },
)
print(r.content)
print(r.status_code)
