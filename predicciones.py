def predict_class(text, model, tokenizer):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    prediction = torch.argmax(logits, dim=1).item()
    
    # Mapear el índice de la predicción a la categoría correspondiente
    categories = ['Quejas', 'Preguntas frecuentes', 'Consultas técnicas', 'Soluciones']
    return categories[prediction]


if __name__ == '__main__':
    predicted_category = predict_class(preprocessed_text, model, tokenizer)
    print(f"La categoría predicha es: {predicted_category}")
