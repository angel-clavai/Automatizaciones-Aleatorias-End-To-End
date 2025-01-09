from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split

# BERT preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-spanish-wwm-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-spanish-wwm-cased', num_labels=4)

# dataset
class ConversationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
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
    
    # predicción a la categoría correspondiente
    categories = ['Quejas', 'Preguntas frecuentes', 'Consultas técnicas', 'Soluciones']
    return categories[prediction]



if __name__ == '__main__':

    texts = ["Tengo un problema con mi factura", "Cómo puedo cambiar mi contraseña", "Mi conexión a internet no funciona", "Quiero hacer una devolución"]
    labels = [0, 1, 2, 3]  # 0: quejas, 1: preguntas frecuentes, 2: consultas técnicas, 3: soluciones

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    train_dataset = ConversationDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = ConversationDataset(val_texts, val_labels, tokenizer, max_len=128)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

