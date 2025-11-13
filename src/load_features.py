from feature_extraction import load_features

train_features = load_features("features/train_multimodal_features.npz")
val_features = load_features("features/val_multimodal_features.npz")
test_features = load_features("features/test_multimodal_features.npz")

audio_train = train_features['audio_features']
text_train = train_features['text_features']

audio_test = test_features['audio_features']
text_test = test_features['text_features']

audio_val = val_features['audio_features']
text_val = val_features['text_features']

print("Audio training features shape: ", audio_train.shape)
print("Text training features shape: ", text_train.shape)

print("Audio validation features shape: ", audio_val.shape)
print("Text validation features shape: ", text_val.shape)

print("Audio test features shape: ", audio_test.shape)
print("Text test features shape: ", text_test.shape)