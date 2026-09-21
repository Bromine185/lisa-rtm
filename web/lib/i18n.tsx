"use client";
// Two languages, one dictionary. Technical tokens (arm names, CRPS, LSD, ViSQOL, τ, λ, kHz) are the
// same in both and are never translated; only the prose around them is. The language is remembered
// in localStorage and mirrored onto <html lang>. Nothing here touches a number.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type Lang = "en" | "ja";

const DICT = {
  // header
  "sub": ["12 kHz → 48 kHz · one 88k-parameter network · everything above 6 kHz is generated",
          "12 kHz → 48 kHz · 8.8万パラメータのネットワーク1つ · 6 kHz より上はすべて生成"],
  "legend.in": ["input band, 0–6 kHz", "入力帯域 0–6 kHz"],
  "legend.gen": ["generated, 6–24 kHz", "生成帯域 6–24 kHz"],
  "legend.truth": ["truth", "原音"],
  "tour": ["tour", "ガイド"],
  "keys": ["keys", "キー操作"],

  // left rail
  "speaker": ["Speaker", "話者"],
  "speaker.hint": ["held out · never trained on", "ホールドアウト · 学習に未使用"],
  "rate": ["Output rate", "出力レート"],
  "rate.hint": ["same weights, queried differently", "同じ重み、問い合わせ方が違うだけ"],
  "model": ["Model", "モデル"],
  "model.hint": ["one init, identical batches", "同一初期化 · 同一バッチ"],
  "readout": ["Readout", "読み出し"],
  "readout.hint": ["one model, three statistics", "1つのモデル、3つの統計量"],
  "draw": ["Draw", "サンプル"],
  "draw.hint": ["τ = 0 is LISA exactly", "τ = 0 は LISA そのもの"],
  "voice": ["Your voice", "あなたの声"],
  "voice.hint": ["record, or drop a file", "録音、またはファイルをドロップ"],

  "ro.draw": ["one draw", "1サンプル"],
  "ro.mean16": ["mean of 16", "16個の平均"],
  "ro.logmean16": ["log-mean of 16", "16個の対数平均"],
  "ro.tau0": ["τ = 0", "τ = 0"],
  "ro.draw.hint": ["a single sample from p(y|x) — what you would ship", "p(y|x) からの1サンプル — 実際に出荷するもの"],
  "ro.mean16.hint": ["the waveform mean — the SNR-optimal readout, and the muffled one", "波形の平均 — SNR 最適だが、こもった音になる読み出し"],
  "ro.logmean16.hint": ["per-bin mean of log|STFT| over 16 draws — LSD's actual minimiser", "16サンプルの log|STFT| をビンごとに平均 — LSD の真の最小化子"],
  "ro.tau0.hint": ["noise off: this is the deterministic LISA exactly", "ノイズ停止：決定論的な LISA そのもの"],
  "ro.live": ["live speaker: one draw only — the statistics need 16 draws", "ライブ話者：1サンプルのみ — 統計量には16サンプルが必要"],

  "tau": ["temperature τ", "温度 τ"],
  "seed": ["seed", "シード"],
  "newdraw": ["new draw", "新しいサンプル"],
  "arch.open.one": ["open the {arm} architecture", "{arm} のアーキテクチャを開く"],

  // model descriptions
  "arm.det_paper": ["LISA as published: plain L1, τ = 0", "論文どおりの LISA：L1 のみ、τ = 0"],
  "arm.det": ["L1 + λ·STFT, τ = 0", "L1 + λ·STFT、τ = 0"],
  "arm.es_marg": ["energy score, log-mag", "エネルギースコア、対数振幅"],
  "arm.es_dec_l0.01": ["noise at the decoder", "デコーダにノイズ"],
  "arm.es_erb_l0.001": ["+ERB, weight too small: collapses toward a point predictor", "+ERB、重みが小さすぎ：点予測へ崩壊"],
  "arm.es_erb_l0.01": ["+ERB: scores log ERB band energies", "+ERB：対数 ERB 帯域エネルギーを採点"],
  "arm.es_erb_l0.1": ["+ERB, spectral weight ×10", "+ERB、スペクトル重み ×10"],
  "arm.es_dec_erb_l0.1": ["both: decoder noise + ERB", "両方：デコーダノイズ + ERB"],
  "kind.plain": ["plain L1 (the paper)", "L1 のみ（論文）"],
  "kind.l1stft": ["L1 + λ·STFT", "L1 + λ·STFT"],
  "kind.es.logmag": ["energy score, log-mag", "エネルギースコア、対数振幅"],
  "kind.es.erb": ["energy score, +ERB", "エネルギースコア、+ERB"],

  // stage
  "view.output": ["output", "出力"],
  "view.truth": ["truth", "原音"],
  "view.input": ["input", "入力"],
  "autorun": ["auto-run", "自動実行"],
  "run": ["run inference", "推論を実行"],
  "stop": ["stop", "停止"],
  "engine.none": ["engine not loaded", "エンジン未読み込み"],
  "src.input": ["input", "入力"],
  "src.output": ["output", "出力"],
  "src.truth": ["truth", "原音"],
  "play": ["play", "再生"],
  "pause": ["pause", "一時停止"],
  "prose": ["Press play, then switch between {input}, {output} and {truth} while it runs — the three are phase-locked, so the only thing that changes is the band above the dashed line. Then change the {readout}: same weights, same draws, a different statistic taken from them.",
            "再生を押し、再生中に{input}・{output}・{truth}を切り替えてください。3つは位相が揃っているので、変わるのは破線より上の帯域だけです。次に{readout}を変えてみてください：同じ重み、同じサンプルから、違う統計量を取り出します。"],

  // status
  "loading": ["loading", "読み込み中"],
  "loading.spk": ["loading {id}", "{id} を読み込み中"],
  "running": ["running", "実行中"],
  "stopped": ["stopped", "停止"],
  "error": ["error", "エラー"],
  "cancelled": ["cancelled", "中断"],
  "noout": ["no output yet — run inference", "出力なし — 推論を実行してください"],
  "audio.in": ["{secs} s audio in {dt} s", "{secs} 秒の音声を {dt} 秒で"],
  "realtime": ["realtime", "実時間"],
  "realtime.here": ["realtime here", "実時間（このブラウザ）"],
  "bench.cpu": ["on an {cpu} CPU, one pass", "{cpu} CPU で、1パス"],
  "bench.the": ["the bench", "ベンチ機"],
  "above24": ["above 24 kHz at ×8", "×8 での 24 kHz 以上"],
  "tag.live": ["live · this browser · τ {tau} · seed {seed}", "ライブ · このブラウザ · τ {tau} · シード {seed}"],
  "tag.pre": ["{lbl} · precomputed · 50-epoch checkpoint", "{lbl} · 事前計算 · 50エポックのチェックポイント"],
  "lbl.tau0": ["τ = 0", "τ = 0"],
  "lbl.draw": ["one draw", "1サンプル"],
  "lbl.mean16": ["mean of 16", "16個の平均"],
  "lbl.logmean16": ["log-mean of 16", "16個の対数平均"],
  "weights.unavailable": ["weights unavailable for {arm}", "{arm} の重みがありません"],
  "manifest.none": ["no audio manifest — run demo/tools/make_fixtures.py", "音声マニフェストなし — demo/tools/make_fixtures.py を実行"],
  "fixtures.none": ["no fixtures", "フィクスチャなし"],

  // numbers rail
  "n.pending": ["evaluation", "評価"],
  "n.pending.v": ["pending", "未完了"],
  "n.pending.note": ["The evaluation cells have not written results.json yet. The audio and the live inference still work.", "評価セルがまだ results.json を書き出していません。音声とライブ推論は動作します。"],
  "h.disease": ["the disease and the cure", "病と治療"],
  "h.judges": ["the judges · with passthrough", "評価指標 · 低域パススルー込み"],
  "h.cost": ["cost · measured, one thread pool", "計算コスト · 実測、1スレッドプール"],
  "h.snr": ["SNR · maximised by doing nothing", "SNR · 何もしないのが最大"],
  "r.deficit.draw": ["deficit · 1 draw", "高域欠損 · 1サンプル"],
  "r.deficit.tau0": ["deficit · τ = 0", "高域欠損 · τ = 0"],
  "r.coherent": ["coherent fraction", "コヒーレント率"],
  "r.kappa": ["κ · hallucinated?", "κ · 幻覚か？"],
  "r.pit": ["PIT end bins", "PIT 端ビン"],
  "r.pit.ideal": ["ideal .118", "理想 .118"],
  "r.lsd.draw": ["LSD · 1 draw", "LSD · 1サンプル"],
  "r.share": ["share of the band", "帯域内の位置"],
  "r.share.d": ["floor→ceiling", "下限→上限"],
  "r.visqol.audio": ["ViSQOL audio", "ViSQOL audio"],
  "r.visqol.speech": ["ViSQOL speech", "ViSQOL speech"],
  "r.pesq": ["PESQ wb", "PESQ wb"],
  "r.params": ["parameters", "パラメータ数"],
  "r.onepass": ["one pass · per s of audio", "1パス · 音声1秒あたり"],
  "r.frame": ["one 20 ms frame", "20 ms フレーム1つ"],
  "r.shipped1": ["shipped · 1 output + pt", "出荷形 · 出力1 + パススルー"],
  "r.shipped16": ["shipped · logmean16 + pt", "出荷形 · logmean16 + パススルー"],
  "r.utt": ["{s} s utt", "{s} 秒の発話"],
  "r.lookahead": ["algorithmic lookahead", "アルゴリズム的先読み"],
  "r.naive": ["naive sinc upsample", "単純 sinc アップサンプル"],
  "r.naive.d": ["naive", "単純比"],
  "r.floor": ["floor · empty high band", "下限 · 高域なし"],
  "r.ceiling": ["ceiling · true high band", "上限 · 真の高域"],
  "n.utts": ["{n} held-out utterances", "ホールドアウト発話 {n} 件"],
  "n.of.one": ["of one speaker ({spk})", "話者1名（{spk}）"],
  "n.of.many": ["of {k} speakers ({spk})", "話者 {k} 名（{spk}）"],
  "n.draws": ["{M} draws", "{M} サンプル"],
  "n.epochs": ["{e} epochs", "{e} エポック"],
  "n.delta": ["Δ vs {ref} ({kind}, τ = 0); det_paper is the paper’s model", "Δ は {ref}（{kind}、τ = 0）比；det_paper は論文のモデル"],
  "n.cost": ["cost on {cpu}, {threads} threads, {precision}, torch {torch}, median of repeats; passthrough alone {ms}", "コストは {cpu}、{threads} スレッド、{precision}、torch {torch}、反復の中央値；パススルーのみ {ms}"],
  "n.bench.cpu": ["the bench CPU", "ベンチ機の CPU"],
  "n.fp32": ["fp32 eager", "fp32 eager"],
  "gate.title": ["High band by frame loudness", "フレーム音量別の高域"],
  "gate.sub": ["energy ratio, dB", "エネルギー比 dB"],
  "gate.loud": ["loud", "大"],
  "gate.mid": ["mid", "中"],
  "gate.quiet": ["quiet", "小"],
  "gate.note": ["A correct sampler shows the same ratio in every row. A ratio that rises as frames get quieter is energy where the truth has none — that is what a listener hears as hiss.",
                "正しいサンプラーはどの行でも同じ比率を示します。フレームが静かになるほど比率が上がるなら、それは原音に存在しないエネルギーで、聴き手にはヒスとして聞こえます。"],
  "arch.open": ["open the architecture", "アーキテクチャを開く"],
  "n.foot": ["Numbers are from the evaluation the run wrote, on {utts}, {draws}{epochs}. Nothing on this page is typed in.",
             "数値はすべてこの実行が書き出した評価結果です（{utts}、{draws}{epochs}）。このページに手入力の数値はありません。"],

  // voice
  "v.record": ["record", "録音"],
  "v.stop": ["stop", "停止"],
  "v.drop": ["drop audio", "音声をドロップ"],
  "v.recording": ["recording · {s} s", "録音中 · {s} 秒"],
  "v.processing": ["processing", "処理中"],
  "v.denied": ["microphone unavailable", "マイクを使用できません"],
  "v.badfile": ["could not decode that file", "そのファイルを読み込めませんでした"],
  "v.short": ["too short — hold for at least a second", "短すぎます — 1秒以上録音してください"],
  "v.note": ["Recorded at 48 kHz, everything above 6 kHz discarded to make the 12 kHz input, then through the network. Training was English only (VCTK); anything else is out of distribution — that is the experiment.",
             "48 kHz で録音し、6 kHz より上を捨てて 12 kHz の入力を作り、ネットワークに通します。学習は英語（VCTK）のみ — それ以外は分布外です。それこそが実験です。"],
  "v.name": ["you", "あなた"],
  "v.meta": ["live · this laptop's microphone", "ライブ · このノートPCのマイク"],
  "v.meta.file": ["live · dropped file", "ライブ · ドロップしたファイル"],

  // blind test
  "b.title": ["Blind test", "ブラインド試聴"],
  "b.start": ["blind test", "ブラインド試聴"],
  "b.exit": ["exit", "終了"],
  "b.q": ["Which sounds more natural?", "どちらが自然に聞こえますか？"],
  "b.hint": ["Play, switch A / B, then vote. The spectrogram is hidden during a trial; only the band above 6 kHz differs.",
             "再生し、A / B を切り替えて投票してください。試行中はスペクトログラムを隠します。違うのは 6 kHz より上の帯域だけです。"],
  "b.vote.a": ["A", "A"],
  "b.vote.b": ["B", "B"],
  "b.vote.tie": ["can't tell", "判別不能"],
  "b.reveal": ["A was {a} · B was {b}", "A は {a} · B は {b}"],
  "b.next": ["next trial", "次の試行"],
  "b.tally": ["tally", "集計"],
  "b.sampler": ["sampler", "サンプラー"],
  "b.det": ["deterministic", "決定論的"],
  "b.tie": ["tie", "引き分け"],
  "b.reset": ["reset", "リセット"],
  "b.trial": ["trial {n}", "試行 {n}"],
  "b.pair": ["{sampler} ({ro}) against {det} (τ = 0), speaker chosen at random each trial", "{sampler}（{ro}）対 {det}（τ = 0）、話者は試行ごとにランダム"],
  "b.preparing": ["preparing the pair", "ペアを準備中"],
  "b.picked": ["you picked {v}", "あなたの選択：{v}"],

  // tour
  "tour.title": ["The Missing Band — a guided tour", "The Missing Band — ガイド"],
  "tour.1.h": ["1 · The problem", "1 · 問題"],
  "tour.1.p": ["LISA (ICASSP 2022) is a tiny audio super-resolution network: 12 kHz in, 48 kHz out, ~88k parameters. Trained on a deterministic loss, it converges to the conditional mean — and the mean of a random-phase high band has no energy. The output is muffled. That is not a bug; it is what the objective asks for.",
               "LISA（ICASSP 2022）は約 8.8 万パラメータの小さな音声超解像ネットワークです：入力 12 kHz、出力 48 kHz。決定論的な損失で学習すると条件付き平均に収束しますが、位相がランダムな高域の平均にはエネルギーがありません。出力はこもります。バグではなく、目的関数がそう求めているのです。"],
  "tour.2.h": ["2 · The fix is an objective, not an architecture", "2 · 解決策はアーキテクチャではなく目的関数"],
  "tour.2.p": ["Concatenate eight Gaussian channels at the input and train the same network under the energy score, two draws per step. The score is strictly proper, so the network converges to p(y|x) rather than its mean. One forward pass is one calibrated sample. With the noise at zero it is LISA exactly (+896 weights).",
               "入力に 8 本のガウス雑音チャネルを連結し、同じネットワークをエネルギースコア（1ステップに2サンプル）で学習します。スコアは厳密に proper なので、ネットワークは平均ではなく p(y|x) に収束します。順伝播 1 回が較正された 1 サンプルです。雑音をゼロにすれば LISA そのもの（+896 重み）です。"],
  "tour.3.h": ["3 · Listen", "3 · 聴く"],
  "tour.3.p": ["Press Space, then switch 1 / 2 / 3 between input, output and truth while it plays. They are phase-locked: the only thing that changes is the band above the dashed line. The spectrogram colours the input band blue and the generated band orange.",
               "スペースで再生し、再生中に 1 / 2 / 3 で入力・出力・原音を切り替えてください。3つは位相が揃っているので、変わるのは破線より上の帯域だけです。スペクトログラムでは入力帯域が青、生成帯域がオレンジです。"],
  "tour.4.h": ["4 · Same weights, three readouts", "4 · 同じ重み、3つの読み出し"],
  "tour.4.p": ["One draw is what you would ship. The waveform mean of 16 draws is the SNR-optimal readout — and the muffled one: that is the regression to the mean, made audible. The log-mean of 16 is what LSD actually minimises. Change the readout and hear each judge's preferred answer.",
               "1サンプルは実際に出荷するもの。16サンプルの波形平均は SNR 最適の読み出しですが、こもった音です — 平均への回帰が耳で聞こえます。16サンプルの対数平均は LSD が本当に最小化するものです。読み出しを切り替えて、各評価指標が好む答えを聴き比べてください。"],
  "tour.5.h": ["5 · Your voice, and a blind test", "5 · あなたの声、そしてブラインド試聴"],
  "tour.5.p": ["Press V and speak: the recording is band-limited to 12 kHz on the spot and pushed through the network in this browser. Training was English only, so Japanese is out of distribution — that is the experiment. Press B for a blind A/B test: the sampler against the deterministic model, speaker chosen at random, your ear as the judge.",
               "V を押して話してください：録音はその場で 12 kHz に帯域制限され、このブラウザ内でネットワークに通されます。学習は英語のみなので日本語は分布外 — それこそが実験です。B でブラインド A/B 試聴：サンプラー対決定論的モデル、話者はランダム、判定はあなたの耳です。"],
  "tour.close": ["close", "閉じる"],
  "tour.next": ["next", "次へ"],
  "tour.prev": ["back", "戻る"],
  "tour.esc": ["Esc closes · ? reopens", "Esc で閉じる · ? で再表示"],

  // keys
  "k.space": ["play / pause", "再生 / 一時停止"],
  "k.123": ["input / output / truth", "入力 / 出力 / 原音"],
  "k.r": ["run / stop inference", "推論の実行 / 停止"],
  "k.n": ["new draw", "新しいサンプル"],
  "k.updown": ["model", "モデル"],
  "k.leftright": ["speaker", "話者"],
  "k.v": ["record your voice", "声を録音"],
  "k.b": ["blind test", "ブラインド試聴"],
  "k.l": ["language", "言語"],
  "k.q": ["tour", "ガイド"],

  // architecture page
  "a.back": ["← instrument", "← 計測画面"],
  "a.stage.0": ["48 kHz → 12 kHz", "48 kHz → 12 kHz"],
  "a.stage.1": ["conv encoder", "畳み込みエンコーダ"],
  "a.stage.2": ["latents", "潜在表現"],
  "a.stage.3": ["decoder MLP", "デコーダ MLP"],
  "a.stage.4": ["48 kHz output", "48 kHz 出力"],
  "a.timeline": ["timeline", "タイムライン"],
  "a.scrub": ["scrub the timeline", "タイムラインを動かす"],
  "a.fit": ["fit all", "全体表示"],
  "a.focus": ["click a stage to fly to it", "ステージをクリックで移動"],
  "a.help": ["drag · orbit    two-finger swipe · travel    pinch or ⌘ scroll · zoom    1–5 · stage    0 · fit all    space · play", "ドラッグ · 回転    2本指スワイプ · 移動    ピンチ / ⌘スクロール · ズーム    1–5 · ステージ    0 · 全体表示    space · 再生"],
  "a.other": ["other arms", "他のモデル"],
  "a.missing": ["arch3d.js not loaded", "arch3d.js が読み込まれていません"],
  "a.f.input": ["input", "入力"],
  "a.f.input.v": ["12 kHz · 11-sample receptive field · 0.9 ms", "12 kHz · 受容野 11 サンプル · 0.9 ms"],
  "a.f.encoder": ["encoder", "エンコーダ"],
  "a.f.encoder.v": ["conv 7 / 3 / 3 / 1 · channels 16 / 32 / 64 / 32 · latent 32", "conv 7 / 3 / 3 / 1 · チャネル 16 / 32 / 64 / 32 · 潜在 32"],
  "a.f.decoder": ["decoder", "デコーダ"],
  "a.f.noise": ["noise", "ノイズ"],
  "a.f.noise.sd": ["8 channels at the encoder input + 4 per output sample at the decoder", "エンコーダ入力に 8 チャネル + デコーダで出力サンプルごとに 4"],
  "a.f.noise.det": ["8 channels at the encoder input, held at zero", "エンコーダ入力に 8 チャネル、ゼロ固定"],
  "a.f.noise.s": ["8 channels at the encoder input", "エンコーダ入力に 8 チャネル"],
  "a.f.params": ["parameters", "パラメータ"],
  "a.f.params.v": ["{n} · LISA is 86,881 · τ = 0 recovers it exactly", "{n} · LISA は 86,881 · τ = 0 で完全に一致"],
} as const;

export type Key = keyof typeof DICT;

const STORE = "lisa-rtm.lang";

function detect(): Lang {
  try {
    const s = localStorage.getItem(STORE);
    if (s === "en" || s === "ja") return s;
  } catch { /* private mode */ }
  return typeof navigator !== "undefined" && /^ja\b/i.test(navigator.language) ? "ja" : "en";
}

interface Ctx { lang: Lang; setLang: (l: Lang) => void; t: (k: Key, vars?: Record<string, string | number>) => string }
const LangCtx = createContext<Ctx>({ lang: "en", setLang: () => {}, t: (k) => DICT[k][0] });

export function fill(s: string, vars?: Record<string, string | number>): string {
  return vars ? s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : s;
}

export function LangProvider({ children }: { children: React.ReactNode }) {
  // the server renders English; the stored choice is applied once on the client so the markup matches
  const [lang, setLangState] = useState<Lang>("en");
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { setLangState(detect()); }, []);
  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dataset.lang = lang;
  }, [lang]);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try { localStorage.setItem(STORE, l); } catch { /* private mode */ }
  }, []);
  const t = useCallback((k: Key, vars?: Record<string, string | number>) => fill(DICT[k][lang === "ja" ? 1 : 0], vars), [lang]);
  const v = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <LangCtx.Provider value={v}>{children}</LangCtx.Provider>;
}

export function useT() { return useContext(LangCtx); }

export function hasKey(k: string): k is Key { return k in DICT; }
