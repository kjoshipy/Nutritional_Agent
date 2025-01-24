import os
import marvin
from marvin.beta.assistants import Assistant, pprint_run

# ------------------- Environment Configuration -------------------
def configure_environment():
    """Set up environment variables and API configuration."""
    os.environ["MARVIN_LOG_LEVEL"] = "DEBUG"
    os.environ["MARVIN_VERBOSE"] = "True"

    # OpenAI API Configuration
    OPENAI_KEY = ""  # Insert your API key here
    OPENAI_MODEL = "gpt-4o"  # Accessible model for the provided key

    marvin.settings.openai.api_key = OPENAI_KEY
    marvin.settings.openai.chat.completions.model = OPENAI_MODEL
    marvin.settings.openai.chat.completions.temperature = 0.0
    return OPENAI_MODEL


# ------------------- Assistant Setup -------------------
def create_food_assistant(model):
    """Create and configure the FoodBot assistant."""
    instructions = """
    You are a nutritional assistant who helps users analyze their food choices and provide actionable recommendations.

    When analyzing an uploaded food image:
    - Identify the food items on the plate.
    - Classify each item into a category (protein, grain, vegetable, fruit).
    - Estimate portion sizes in grams.
    - Recommend improvements for the meal based on nutrition science.
    - Focus only on nutrition-related questions.
    """

    # Define food extraction tool
    def food_extractor(path: str):
        img = marvin.Image.from_path(path)
        return marvin.extract(
            img,
            target=str,
            instructions="Identify food items, classify them, estimate portions in grams, and recommend meal improvements.",
        )

    return Assistant(
        name="FoodBot",
        model=model,
        instructions=instructions,
        tools=[food_extractor],
    )


# ------------------- User Contexts -------------------
def get_user_contexts():
    """Define different contexts for testing the assistant."""
    return [
        "Answer as if the user follows a low FODMAP diet for managing irritable bowel syndrome (IBS).",
        "Answer as if the user has celiac disease and requires a strict gluten-free diet.",
        "Answer as if the user follows a ketogenic diet and needs to limit carbohydrate intake.",
        "Answer as if the user follows a DASH diet for managing high blood pressure.",
        "Answer as if the user follows a DASH diet for managing high blood pressure."
    ]


# ------------------- Test Runner -------------------
def run_tests(food_assistant, user_contexts):
    """Run test cases with various user contexts."""
    for context in user_contexts:
        print(f"\n--- Testing Context: {context} ---")
        upload_message = f"I'd like to upload an image of my recent meal for you to analyze. {context}"
        analysis_message = "Here is the path for the food image to analyze: 'enchilada-image.jpg'"

        response1 = food_assistant.say(upload_message)
        response2 = food_assistant.say(analysis_message)

        pprint_run(response1)
        pprint_run(response2)


# ------------------- Main Execution -------------------
if __name__ == "__main__":
    configure_environment()
    openai_model = configure_environment()
    food_bot = create_food_assistant(openai_model)
    user_contexts = get_user_contexts()
    run_tests(food_bot, user_contexts)
