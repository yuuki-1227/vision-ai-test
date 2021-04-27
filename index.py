from google.cloud import vision

# with 開始から終了まで自動で実行してくれる
# rb read binaryモード　バイナリーモードを読み込む
# テキスト以外のデータ　主に画像や動画

# road.jpgを開いて読み込む
with open('./road.jpg', 'rb') as image_file:
    content = image_file.read()

# vision APIが扱える画像データに変換
image = vision.Image(content=content)

# annotation テキストや音声、画像などあらゆる形式のデータにタグ付けをする作業
# client データを扱う人、もの
# ImageAnnotatorClientのインスタンスを生成
annotater_client = vision.ImageAnnotatorClient()

response_data = annotater_client.label_detection(image=image)
labels = response_data.label_annotations

print('----RESULT----')
for label in labels:
    print(label.description, ':', round(label.score * 100, 2), '%')
print('----RESULT----')
