from feature_extraction import load_features

train_features = load_features("features/train_multimodal_features.npz")
val_features = load_features("features/val_multimodal_features.npz")
test_features = load_features("features/test_multimodal_features.npz")

audio_train = train_features['audio_features']
text_train = train_features['text_features']

print("Audio training features shape: ", audio_train.shape)
print("Text training features shape: ", text_train.shape)