from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("/data/zhuqin/bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("/data/zhuqin/Classfication/outputs5/best_model/")



pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


# texts = ["sent1", "sent2"]

texts = ["Lenovo unique function/Option/App (Vantage, Dock etc.) Major function no work Device detach/attach FR: 2/2 units, 6/6 Trials [SEP] [Lynx 2][Mobile][Win11 unique] 21H2 SIT1.1 22000.434:[TBT4 Dock] System not waking up via WOL from S4/S5 when PCIe tunneling is enabled.(FR: 2/2 units, 6/6 Trials) 1. Preload Lynx2 21H2 SIT1.1 (OS build: win11:22000.434) Win11PRO64EN-UK with BIOS R1ZET26W/EC R1ZHT26W. 2. Boot to BIOS, enable OS Optimised, Kernel DMA. 3. Ensure PCIe Tunneling is enabled, save and exit. 4. Boot to OS, attach TBT4 Dock with LAN to system. 5. Open device manager, notice that Dock LAN is Intel -> PASS 6. Put system to S4/S5 and wake through magic packet with Dock MAC Address. 7. Notice system does not wake.==>PROBLEM A) Unique to a certain system? No - Happen on all EUTs B) Unique to a configuration? No - Happen on all Configs C) Unique to OS? Yes - Happen on Win11 OS D) Unique to a product? Yes - Happen on Lynx2 SIT machines E) Unique to a current version? Yes - Happen on BIOS: R1ZET26W"]

res = pipe(texts)

label2name = {
    "LABEL_0": 0,
    "LABEL_1": 1
}

print(label2name[(res[0])["label"]])

for result in res:
    print("预测的类别为：{}，score为：{}".format(label2name[result["label"]], result["score"]))
