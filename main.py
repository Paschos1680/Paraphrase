import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from typing import Tuple, Optional

st.set_page_config(page_title="Paraphrase Tool")

# Constants
WORD_LIMIT = 200


class ModelManager:
    PARAPHRASE_MODELS = {"BART": "Models/FinalBART", "T5": "Models/saved_t5_model"}

    DETECTION_MODELS = {
        "BERT": "Models/final_model_bert",
        "XLNet": "Models/paraphrase_xlnet_model",
    }

    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split())

    @staticmethod
    def validate_input(text: str) -> Tuple[bool, str]:
        if not text.strip():
            return False, "Input text cannot be empty."
        word_count = ModelManager.count_words(text)
        if word_count > WORD_LIMIT:
            return (
                False,
                f"Input exceeds {WORD_LIMIT} words limit. Current count: {word_count}",
            )
        return True, ""

    @staticmethod
    @st.cache_resource
    def load_model(model_name: str, task: str):
        """Load and cache model"""
        try:
            if task == "paraphrase":
                model_path = ModelManager.PARAPHRASE_MODELS[model_name]
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path
                )  # Changed this line
                return tokenizer, model
            else:  # detection
                model_path = ModelManager.DETECTION_MODELS[model_name]
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                return tokenizer, model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None

    @staticmethod
    def generate_paraphrase(text: str, model_name: str) -> Optional[str]:
        tokenizer, model = ModelManager.load_model(model_name, "paraphrase")
        if not tokenizer or not model:
            return None

        try:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.8,
                top_p=0.8,
                do_sample=True,
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            st.error(f"Error during paraphrasing: {str(e)}")
            return None

    @staticmethod
    def detect_paraphrase(
        text1: str, text2: str, model_name: str
    ) -> Tuple[bool, float]:
        tokenizer, model = ModelManager.load_model(model_name, "detection")
        if not tokenizer or not model:
            return False, 0.0

        try:
            inputs = tokenizer(
                text1, text2, return_tensors="pt", truncation=True, padding=True
            )
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # prob = probabilities[0][1].item()
            # is_paraphrase = bool(prob > 0.5)
            # confidence = prob if is_paraphrase else 1 - prob
            # return is_paraphrase, confidence
            predicted_class_id = probabilities.argmax(dim=1)
            return predicted_class_id == 1, probabilities[0][predicted_class_id].item()
        except Exception as e:
            st.error(f"Error during paraphrase detection: {str(e)}")
            return False, 0.0


def main():
    st.title("Text Paraphrasing Tool")

    # Usecase selection
    usecase = st.radio("Select Use Case", ["Paraphrase Text", "Detect Paraphrase"])

    if usecase == "Paraphrase Text":
        model_name = st.selectbox("Select Model", ["BART", "T5"])

        text = st.text_area(
            "Enter text to paraphrase", help=f"Maximum {WORD_LIMIT} words"
        )

        # Add word count display
        word_count = ModelManager.count_words(text)
        st.caption(f"Word count: {word_count}/{WORD_LIMIT}")

        if st.button("Generate Paraphrase"):
            is_valid, error_message = ModelManager.validate_input(text)

            if not is_valid:
                st.error(error_message)
            else:
                with st.spinner("Generating paraphrase..."):
                    result = ModelManager.generate_paraphrase(text, model_name)
                    if result:
                        st.success("Paraphrase generated successfully!")
                        st.write(result)

    else:  # Detect Paraphrase
        model_name = st.selectbox("Select Model", ["BERT", "XLNet"])

        text1 = st.text_area("Enter first text", help=f"Maximum {WORD_LIMIT} words")
        # Add word count display for first text
        word_count1 = ModelManager.count_words(text1)
        st.caption(f"Word count (Text 1): {word_count1}/{WORD_LIMIT}")

        text2 = st.text_area("Enter second text", help=f"Maximum {WORD_LIMIT} words")
        # Add word count display for second text
        word_count2 = ModelManager.count_words(text2)
        st.caption(f"Word count (Text 2): {word_count2}/{WORD_LIMIT}")

        if st.button("Check Paraphrase"):
            is_valid1, error_message1 = ModelManager.validate_input(text1)
            is_valid2, error_message2 = ModelManager.validate_input(text2)

            if not is_valid1:
                st.error(f"First text: {error_message1}")
            elif not is_valid2:
                st.error(f"Second text: {error_message2}")
            else:
                with st.spinner("Analyzing texts..."):
                    is_paraphrase, confidence = ModelManager.detect_paraphrase(
                        text1, text2, model_name
                    )
                    st.success("Analysis complete!")
                    st.write(
                        f"Result: {'Paraphrase' if is_paraphrase else 'Not Paraphrase'}"
                    )
                    st.write(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
