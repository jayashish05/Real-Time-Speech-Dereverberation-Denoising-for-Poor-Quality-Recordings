"""Build report/Project_Report.pdf (edit the text below, then run: python scripts/build_report.py)."""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
OUT = os.path.join(ROOT, "report", "Project_Report.pdf")

F = "/System/Library/Fonts/Supplemental/Times New Roman"
if os.path.exists(F + ".ttf"):
    for name, suffix in [("Body", ""), ("Body-Bold", " Bold"), ("Body-Italic", " Italic")]:
        pdfmetrics.registerFont(TTFont(name, F + suffix + ".ttf"))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")
    R, B, I = "Body", "Body-Bold", "Body-Italic"
else:  # other platforms
    R, B, I = "Times-Roman", "Times-Bold", "Times-Italic"

body = ParagraphStyle("body", fontName=R, fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
h1 = ParagraphStyle("h1", fontName=B, fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName=B, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", fontName=B, fontSize=18, leading=23, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("sub", fontName=R, fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=12)
cap = ParagraphStyle("cap", fontName=I, fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=10)
cell = ParagraphStyle("cell", fontName=R, fontSize=9, leading=11)
cellb = ParagraphStyle("cellb", parent=cell, fontName=B)
ref = ParagraphStyle("ref", fontName=R, fontSize=9.5, leading=12.5, leftIndent=18, firstLineIndent=-18, spaceAfter=3)

S = []


def P(t):
    S.append(Paragraph(t, body))


def H1(t):
    S.append(Paragraph(t, h1))


def H2(t):
    S.append(Paragraph(t, h2))


def table(rows, widths, caption):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9e9e9")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    S.append(KeepTogether([Paragraph(caption, cap), t, Spacer(1, 10)]))


def figure(name, width_cm, ratio, caption):
    S.append(KeepTogether([Image(os.path.join(FIG, name), width=width_cm * cm, height=width_cm * ratio * cm),
                           Paragraph(caption, cap)]))


# ---------------------------------------------------------------- front matter
S.append(Paragraph("Real-Time Speech Dereverberation and Denoising for Poor-Quality Recordings", title))
S.append(Paragraph("SLP final project report", sub))

H1("Video demo")
box = ParagraphStyle("box", fontName=B, fontSize=13, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
small = ParagraphStyle("small", fontName=R, fontSize=9, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#777777"))
vids = Table([[[Paragraph(f"Video {i}", box), Spacer(1, 4), Paragraph("placeholder", small)] for i in (1, 2, 3)]],
             colWidths=[5.3 * cm] * 3, rowHeights=[3.4 * cm])
vids.setStyle(TableStyle([("BOX", (i, 0), (i, 0), 1, colors.HexColor("#777777"), None, (4, 3)) for i in range(3)] +
                         [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                          ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f6f6"))]))
S.append(vids)
S.append(Spacer(1, 4))
S.append(Paragraph("Space reserved for the three demo videos.", cap))

# ---------------------------------------------------------------- 1
H1("1. Introduction")
P("Speech recorded outside a studio is rarely clean. A lecture captured on a phone picks up the ceiling fan and "
  "the echo of the hall. A call made from a kitchen comes with running water and clattering dishes. People find "
  "such audio tiring to listen to, and speech recognisers make many more mistakes on it.")
P("Most enhancement tools deal with one problem at a time, and that problem is usually noise. In real recordings "
  "noise and reverberation arrive together. Our aim was one system that handles both, and that is light enough to "
  "run on a laptop while the person is still speaking.")
P("We built that system and an application around it. The model is a small causal neural network of about 1.5 "
  "million parameters, which we call DA-CRN (degradation-aware convolutional recurrent network). Apart from "
  "cleaning the audio, it estimates how noisy and how reverberant the input is and shows this to the user. We "
  "trained it, together with eight other models, on speech that we degraded ourselves from public datasets. We "
  "then compared all of them with classical signal processing on quality, intelligibility, recognition accuracy "
  "and speed.")
P("In short, noise removal works well and the system runs in real time with a 32 ms delay. Removing "
  "reverberation works only a little. Both outcomes are reported in Section 4 with the numbers we measured.")

# ---------------------------------------------------------------- 2
H1("2. Literature review")
H2("2.1 Classical methods")
P("The oldest family of methods estimates the noise spectrum and turns down the frequency bins where noise "
  "dominates. The Wiener filter and the MMSE estimator of Ephraim and Malah [1] are the standard examples. They "
  "are cheap and behave well with steady noise such as a fan, but they cannot follow noise that changes quickly. "
  "For reverberation the usual classical choice is weighted prediction error (WPE) [2], which predicts the late "
  "echo from past frames and subtracts it. A Python implementation is available as NARA-WPE [3]. WPE does not "
  "remove additive noise.")
H2("2.2 Neural networks that run in real time")
P("Tan and Wang [4] showed that a convolutional encoder, a recurrent layer and a decoder can enhance speech "
  "frame by frame without looking at future audio. DCCRN [5] extended this design with a complex mask so that "
  "the phase is corrected as well as the magnitude, and it did well in the first Deep Noise Suppression (DNS) "
  "challenge [15]. NSNet2 [6], the DNS baseline, is a simpler recurrent network that predicts a magnitude mask. "
  "More recent work reduces the cost further: DeepFilterNet [7] reports a real-time factor of 0.19 on one laptop "
  "thread, and GTCRN [8] needs only 23.7 thousand parameters.")
P("Other families exist but fit our problem less well. Conv-TasNet [9] works on the raw waveform and was "
  "designed for separating speakers. MetricGAN+ [10] is trained to maximise the PESQ score directly, but it uses "
  "bidirectional layers, so it needs the whole recording before it can start.")
H2("2.3 Noise and reverberation together")
P("Zhao, Wang and Wang [11] handle the joint problem with two networks in a row, one for noise and one for "
  "reverberation. The common benchmarks for this setting are WHAMR! [17] and the REVERB challenge [16]. Both are "
  "built on speech corpora that require a paid licence. The DNS challenge data is free but takes roughly a "
  "terabyte, and the popular VoiceBank+DEMAND set [18] contains no reverberation. The URGENT challenge [28] goes "
  "one step further and treats noise, reverberation, clipping and limited bandwidth as a single task.")
H2("2.4 Telling the network about the recording condition")
P("Seltzer, Yu and Wang [13] found that a recogniser copes better with noise when a noise estimate is supplied "
  "as an extra input. Wu and colleagues [12] did something similar for reverberation: they estimate the "
  "reverberation time and pick the network settings to match. Their system handles reverberation only and works "
  "offline. FiLM [14] is a general way to let a side input scale and shift the features inside a network.")
P("We did not find a compact streaming model that estimates both the noise level and the reverberation and "
  "uses these estimates to adjust its own processing. That is the idea we set out to test.")

# ---------------------------------------------------------------- 3
H1("3. Methodology")
H2("3.1 Data")
P("We could not use the large benchmarks, for the reasons given above, so we created the degraded speech "
  "ourselves from three free corpora (Table 1). Clean speech comes from LibriSpeech [19], which also provides "
  "transcripts, so we can measure recognition errors. Room responses and noise come from the OpenSLR SLR28 "
  "collection [20]. Noise from DEMAND [21] is kept aside for testing only.")
table([["Role", "Source", "Amount used"],
       ["Clean speech, training", "LibriSpeech train-clean-100 (subset)", "40 speakers, 4,322 utterances, 15.6 hours"],
       ["Clean speech, validation", "LibriSpeech dev-clean", "40 speakers, 5.4 hours"],
       ["Clean speech, test", "LibriSpeech test-clean", "40 speakers; 100 utterances of 3 to 10 s"],
       ["Room responses", "SLR28 simulated rooms", "6,438 training, 807 validation, 785 test"],
       ["Room responses, unseen", "SLR28 measured rooms", "227 real rooms, test only"],
       ["Noise", "SLR28 point-source noises", "672 training files (4.8 h), 75 validation, 79 test"],
       ["Noise, unseen", "DEMAND, 4 environments", "living room, office, cafeteria, bus; test only"]],
      [4.2, 5.6, 6.6], "Table 1. Data used in the project.")
P("The training speakers, rooms and noise files never appear in the test sets. We planned to use all 100 hours "
  "of LibriSpeech training speech, but our internet connection was slow, so we trained on the first 40 speakers.")
H2("3.2 Creating the degraded speech")
P("Each training example is produced on the fly. A 4 second piece of clean speech is passed through a room "
  "response to add reverberation, and noise is then added at a level drawn between -5 and 20 dB signal-to-noise "
  "ratio. Ten percent of the examples stay clean, 30 percent get noise only, 20 percent reverberation only and "
  "40 percent both. In one example out of ten we also clip the waveform, to imitate an overloaded microphone. We "
  "only kept rooms with a reverberation time (T60) of at most 1.5 seconds, because some simulated rooms in SLR28 "
  "reach 7 seconds, which no normal room does.")
P("Since we create the degradation, we also know the correct answer for each example: the clean speech as it "
  "arrives on the direct path, without echo. We know the noise level and the reverberation time as well, and "
  "these serve as labels for the part of the model that analyses the recording.")
P("For testing we fixed nine sets of 100 utterances. The same 100 sentences appear in every set, so the systems "
  "can be compared sentence by sentence. Five sets match the training conditions: clean, noise, reverberation, "
  "both, and a severe case. Four sets contain something the models never saw: DEMAND noise, measured rooms, noise "
  "levels below the training range, and a telephone channel.")
H2("3.3 The model")
P("The signal is cut into 32 ms frames with a 16 ms step and converted to a spectrum. Five convolution layers "
  "compress each frame, two recurrent (GRU) layers follow how the sound develops over time, and five decoding "
  "layers produce a mask. Multiplying the noisy spectrum by this mask gives the enhanced spectrum, which is "
  "converted back to audio. No layer looks at future frames, so the delay is one frame length, 32 ms.")
P("The part we added is a small branch that reads the compressed features and estimates three quantities every "
  "16 ms: the signal-to-noise ratio, the reverberation time T60 and the clarity index C50. The estimates are fed "
  "back into the network through FiLM layers [14], which scale and shift the features before the recurrent "
  "layers. The same estimates are shown to the user as a short report, for example \"noise and reverberation, "
  "severe\".")
H2("3.4 Training")
P("The loss has three terms. The first is the scale-invariant signal-to-distortion ratio [22], which compares "
  "waveforms. The second is a multi-resolution spectral loss [23], which compares spectrograms at three window "
  "sizes. The third is the error of the three estimates. Every model was trained for 6 epochs of 500 steps with "
  "16 clips per step, using the AdamW optimiser on an Apple M2 laptop. One model takes between 35 and 65 minutes. All "
  "models received the same budget so that the comparison is fair.")
H2("3.5 Systems compared")
P("We compared the unprocessed input, three classical systems (Wiener filter, WPE, and WPE followed by Wiener), "
  "an NSNet2-style network [6] that we trained on our data, the plain network without the estimating branch "
  "(called CRN below), and DA-CRN. To see which design choices help, we trained six more variants: the plain "
  "network on noise-only data and on reverberation-only data, one without the spectral loss, one with the "
  "estimating branch but without FiLM, one whose FiLM layers receive the true labels in place of estimates, and a "
  "repeat of DA-CRN with identical settings.")
H2("3.6 Evaluation")
P("Quality is measured with PESQ [24] and with DNSMOS [26], which needs no clean reference. Intelligibility is "
  "measured with STOI [25], and distortion with SI-SDR. For recognition accuracy we pass the audio to the "
  "wav2vec 2.0 recogniser [27] and count the word error rate (WER). Because every system processes the same "
  "sentences, we tested differences with a paired bootstrap over the 400 degraded test utterances.")
H2("3.7 Real-time engine and application")
P("For live use the model processes one 16 ms frame at a time and keeps its internal state between frames. An "
  "automated test confirms that this gives the same output as processing the whole file at once. The "
  "application is written in Streamlit. A user can upload a recording or speak into the microphone, listen to "
  "the audio before and after, compare waveforms and spectrograms, and read the scores and the degradation "
  "report.")

# ---------------------------------------------------------------- 4
H1("4. Results")
H2("4.1 Main comparison")
P("Table 2 gives the scores on the test set that contains both noise and reverberation. The average noise level "
  "of this set is 8.5 dB (signal-to-noise ratio) and the average reverberation time is 0.53 seconds.")
table([["System", "PESQ", "STOI", "SI-SDR (dB)", "WER (%)", "DNSMOS"],
       ["Unprocessed", "1.277", "0.782", "-0.46", "25.7", "1.69"],
       ["Wiener filter", "1.300", "0.771", "-0.10", "25.2", "1.87"],
       ["WPE", "1.292", "0.788", "-0.26", "23.8", "1.74"],
       ["WPE + Wiener", "1.315", "0.778", "0.02", "23.8", "1.93"],
       ["NSNet2-style", "1.392", "0.792", "0.05", "25.4", "2.05"],
       ["CRN (no estimating branch)", "1.448", "0.803", "1.06", "23.2", "2.26"],
       ["DA-CRN (ours)", "1.456", "0.801", "1.09", "24.4", "2.31"],
       ["Clean target (upper limit)", "4.644", "1.000", "n/a", "3.0", "3.22"]],
      [5.4, 2.0, 2.0, 2.5, 2.2, 2.3],
      "Table 2. Results on the noise + reverberation test set (100 utterances from unseen speakers). "
      "Higher is better except for WER.")
P("The neural models give better quality than the classical ones on every degraded set (Figure 1). Averaged over "
  "the four degraded sets, DA-CRN is ahead of WPE + Wiener by 0.12 PESQ and 1.7 dB SI-SDR, and both differences "
  "are statistically significant.")
figure("fig07_pesq.png", 16, 576 / 1600, "Figure 1. PESQ of each system on the five matched test conditions.")
P("Noise is where the system is strongest. On the noise-only set DA-CRN raises PESQ from 1.54 to 1.85 and "
  "reduces the word error rate from 20.6 to 15.3 percent. Reverberation is a different story. On the "
  "reverberation-only set PESQ moves from 1.97 to 2.07, and plain WPE reaches 2.11. Worse, the recogniser makes "
  "more errors after our processing (10.1 percent against 7.0 percent before), whereas WPE lowers the error rate "
  "to 5.8 percent. The network appears to leave small artefacts that listeners and quality scores tolerate but "
  "the recogniser does not.")
figure("fig02_spectrograms.png", 11, 927 / 1043,
       "Figure 2. Spectrograms of one test utterance: degraded input, DA-CRN output and clean target.")
H2("4.2 What each design choice contributes")
table([["Variant", "PESQ", "STOI", "SI-SDR (dB)", "WER (%)"],
       ["CRN, noise-only training data", "1.608", "0.815", "2.84", "24.1"],
       ["CRN, reverberation-only training data", "1.524", "0.805", "0.85", "26.8"],
       ["CRN, both (reference)", "1.624", "0.816", "3.03", "26.4"],
       ["CRN without the spectral loss", "1.547", "0.805", "2.17", "26.1"],
       ["Estimating branch, no FiLM", "1.663", "0.825", "3.09", "23.7"],
       ["DA-CRN", "1.639", "0.813", "3.17", "26.7"],
       ["DA-CRN, repeated run", "1.669", "0.825", "3.31", "24.2"],
       ["FiLM with true labels", "1.677", "0.823", "3.13", "24.0"]],
      [7.0, 2.1, 2.1, 2.7, 2.5],
      "Table 3. Ablation results, averaged over the four degraded test sets (400 utterances).")
P("Three things can be read from Table 3. The spectral loss matters: without it PESQ drops by 0.076 and SI-SDR "
  "by 0.86 dB. The training data must contain both kinds of degradation, since the model trained on "
  "reverberation only is the weakest. And every variant with the estimating branch scores higher on PESQ than "
  "the plain CRN, by 0.016 to 0.054, which is small but statistically significant.")
P("The result for our main idea is less clear. The variant that has the estimating branch but no FiLM layers "
  "does as well as the variants with FiLM. So the gain seems to come from training the network on the extra "
  "estimation task, and feeding the estimates back did not add to it. We also found that two runs with identical "
  "settings differ by 0.029 PESQ and 2.6 points of WER. That is as large as the differences between the methods, "
  "so comparisons based on a single run each should be read with care.")
H2("4.3 Conditions not seen in training")
P("With DEMAND noise, which comes from a different corpus, PESQ still improves (1.48 to 1.58), although the "
  "error rate of the recogniser rises from 20.3 to 24.1 percent. With measured rooms the gain is small (1.20 to "
  "1.28). At noise levels below the training range the model holds up well and reduces the word error rate from "
  "67.9 to 59.8 percent. The telephone channel is a failure case: there is almost no gain, and recognition gets "
  "worse. Figure 3 shows how the benefit depends on the input noise level.")
figure("fig17_pesq_vs_snr.png", 9.5, 544 / 832,
       "Figure 3. PESQ against input signal-to-noise ratio for four systems.")
H2("4.4 Speed and the degradation report")
P("On one CPU thread of the M2, DA-CRN needs 0.60 ms to process a 16 ms frame, a real-time factor of 0.037. In a "
  "live test with the built-in microphone the total delay from input to output was about 113 ms, of which 32 ms "
  "is due to the algorithm and the rest to audio buffering, and no audio block arrived late.")
P("The estimating branch has an average error of 6.0 dB for the noise level and 0.24 seconds for the "
  "reverberation time. It identifies the type of degradation (clean, noise, reverberation or both) correctly for "
  "81 percent of the test recordings. Most of its mistakes are recordings with both problems that it labels as "
  "noise only.")

# ---------------------------------------------------------------- 5
H1("5. Conclusion")
P("We built a working speech enhancement system that runs in real time on a laptop CPU, together with an "
  "application that lets a user try it on files or on live speech. A network of 1.5 million parameters, trained "
  "for about an hour on simulated data, improves perceived quality by a significant margin and outperforms the "
  "classical Wiener and WPE methods on PESQ, STOI and SI-SDR.")
P("The system is good at removing noise and weak at removing reverberation. For reverberant speech that will be "
  "sent to a recogniser, classical WPE remains the better choice in our tests.")
P("Our own addition, the degradation estimate, gave a small improvement and a readable report for the user. The "
  "second half of the idea, letting that estimate steer the network, did not show a benefit of its own with the "
  "training time we had.")

# ---------------------------------------------------------------- 6
H1("6. Future work")
P("The first thing we would do is train each configuration several times with different random seeds and for "
  "longer. All models were still improving when we stopped, and the spread between two identical runs was too "
  "large to settle the FiLM question.")
P("For reverberation, we would change the training target so that the early reflections are kept and only the "
  "late echo is removed. This is an easier task and should produce fewer artefacts. It may also help to train "
  "with a recogniser in the loop so that the output suits it better.")
P("The telephone result shows that the model needs to see band-limited and compressed audio during training. "
  "Adding these, along with the full 100 hours of LibriSpeech and more noise types, is a direct extension. We "
  "would also like to test on real recordings made in classrooms and to run a small listening test with human "
  "listeners, because all our quality scores are computed by algorithms.")

# ---------------------------------------------------------------- 7
H1("References")
REFS = [
    'Y. Ephraim and D. Malah, "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator," IEEE Trans. Acoustics, Speech, and Signal Processing, vol. 32, no. 6, 1984. doi:10.1109/TASSP.1984.1164453',
    'T. Nakatani et al., "Speech dereverberation based on variance-normalized delayed linear prediction," IEEE Trans. Audio, Speech, and Language Processing, vol. 18, no. 7, 2010. doi:10.1109/TASL.2010.2052251',
    'L. Drude, J. Heymann, C. Boeddeker and R. Haeb-Umbach, "NARA-WPE: A Python package for weighted prediction error dereverberation in Numpy and Tensorflow for online and offline processing," ITG Conference on Speech Communication, 2018.',
    'K. Tan and D. Wang, "A convolutional recurrent neural network for real-time speech enhancement," Interspeech, 2018. doi:10.21437/Interspeech.2018-1405',
    'Y. Hu et al., "DCCRN: Deep complex convolution recurrent network for phase-aware speech enhancement," Interspeech, 2020. arXiv:2008.00264',
    'S. Braun and I. Tashev, "Data augmentation and loss normalization for deep noise suppression," SPECOM, 2020. arXiv:2008.06412',
    'H. Schröter, T. Rosenkranz, A. N. Escalante-B. and A. Maier, "DeepFilterNet: Perceptually motivated real-time speech enhancement," Interspeech, 2023. arXiv:2305.08227',
    'X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu and J. Lu, "GTCRN: A speech enhancement model requiring ultralow computational resources," ICASSP, 2024. doi:10.1109/ICASSP48485.2024.10448310',
    'Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation," IEEE/ACM Trans. Audio, Speech, and Language Processing, 2019. arXiv:1809.07454',
    'S.-W. Fu et al., "MetricGAN+: An improved version of MetricGAN for speech enhancement," Interspeech, 2021. arXiv:2104.03538',
    'Y. Zhao, Z.-Q. Wang and D. Wang, "Two-stage deep learning for noisy-reverberant speech enhancement," IEEE/ACM Trans. Audio, Speech, and Language Processing, vol. 27, no. 1, 2019. doi:10.1109/TASLP.2018.2870725',
    'B. Wu, K. Li, M. Yang and C.-H. Lee, "A reverberation-time-aware approach to speech dereverberation based on deep neural networks," IEEE/ACM Trans. Audio, Speech, and Language Processing, vol. 25, no. 1, 2017. doi:10.1109/TASLP.2016.2623559',
    'M. Seltzer, D. Yu and Y. Wang, "An investigation of deep neural networks for noise robust speech recognition," ICASSP, 2013. doi:10.1109/ICASSP.2013.6639100',
    'E. Perez et al., "FiLM: Visual reasoning with a general conditioning layer," AAAI, 2018. arXiv:1709.07871',
    'C. K. A. Reddy et al., "The INTERSPEECH 2020 deep noise suppression challenge: Datasets, subjective testing framework, and challenge results," Interspeech, 2020. arXiv:2005.13981',
    'K. Kinoshita et al., "A summary of the REVERB challenge: State-of-the-art and remaining challenges in reverberant speech processing research," EURASIP Journal on Advances in Signal Processing, 2016. doi:10.1186/s13634-016-0306-6',
    'M. Maciejewski et al., "WHAMR!: Noisy and reverberant single-channel speech separation," ICASSP, 2020. arXiv:1910.10279',
    'C. Valentini-Botinhao et al., "Investigating RNN-based speech enhancement methods for noise-robust text-to-speech," ISCA Speech Synthesis Workshop, 2016. doi:10.21437/SSW.2016-24',
    'V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "Librispeech: An ASR corpus based on public domain audio books," ICASSP, 2015. doi:10.1109/ICASSP.2015.7178964',
    'T. Ko et al., "A study on data augmentation of reverberant speech for robust speech recognition," ICASSP, 2017. doi:10.1109/ICASSP.2017.7953152',
    'J. Thiemann, N. Ito and E. Vincent, "The diverse environments multi-channel acoustic noise database (DEMAND)," Proceedings of Meetings on Acoustics, vol. 19, 2013. doi:10.1121/1.4799597',
    'J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey, "SDR: half-baked or well done?" ICASSP, 2019. arXiv:1811.02508',
    'R. Yamamoto, E. Song and J.-M. Kim, "Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram," ICASSP, 2020. arXiv:1910.11480',
    'A. W. Rix, J. G. Beerends, M. P. Hollier and A. P. Hekstra, "Perceptual evaluation of speech quality (PESQ): a new method for speech quality assessment of telephone networks and codecs," ICASSP, 2001. doi:10.1109/ICASSP.2001.941023',
    'C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Trans. Audio, Speech, and Language Processing, vol. 19, no. 7, 2011. doi:10.1109/TASL.2011.2114881',
    'C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors," ICASSP, 2022. arXiv:2110.01763',
    'A. Baevski, H. Zhou, A. Mohamed and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," NeurIPS, 2020. arXiv:2006.11477',
    'W. Zhang et al., "URGENT challenge: Universality, robustness, and generalizability for speech enhancement," Interspeech, 2024. arXiv:2406.04660',
]
for i, r in enumerate(REFS, 1):
    S.append(Paragraph(f"[{i}] {r}", ref))


def page_number(canvas, doc):
    canvas.setFont(R, 9)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))


os.makedirs(os.path.dirname(OUT), exist_ok=True)
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.2 * cm,
                  bottomMargin=2.2 * cm, title="Real-Time Speech Dereverberation and Denoising").build(
    S, onFirstPage=page_number, onLaterPages=page_number)
print("wrote", OUT)
