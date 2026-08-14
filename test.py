from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent
# res = tavily_search("best hotel in india")
# print(res)

# res= search_flights("Plan a 7 days japan trip from Delhi")
# print(res)

user_input = input("Enter travel Request :")
response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user"
)


print("\n FINAL RESPONSE :\n")
print(response["answer"])

