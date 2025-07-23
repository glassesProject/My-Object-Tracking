from pydub import AudioSegment

audio= AudioSegment.from_file("Audio/beep.wav", format="wav")

# 最後の3秒間でフェードアウト
audio_with_fade_out = audio.fade_out(10)

# # フェードインとフェードアウトを組み合わせる
# audio_with_fades = audio.fade_in(5000).fade_out(3000)

audio_with_fade_out.export("Audio/beep_fades.wav", format="wav")