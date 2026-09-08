#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small Tkinter front-end for the existing Indian Folk Art AI backend.

The backend modules are kept intact.  The UI only coordinates them.
"""
import subprocess
import math
import logging
import queue
import threading
import time
import wave
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

import sounddevice as sd

import main as backend
import speech
import expert
import tts
import vision


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
class Theme:
    BG = "#F5EBDD"
    PANEL = "#FBF6EC"
    CARD = "#FFFDF7"
    CREAM = "#FFF7EA"
    BROWN = "#3A2413"
    BROWN_LIGHT = "#6B4A2F"
    MUTED = "#8A735B"
    TERRACOTTA = "#B6532D"
    TERRACOTTA_DARK = "#873D20"
    ORANGE = "#DC722C"
    GOLD = "#C89532"
    GREEN = "#5D7A45"
    LINE = "#DCC9A8"
    SHADOW = "#E0D0B5"
    USER_BG = "#B6532D"
    USER_FG = "#FFF8ED"
    BOT_BG = "#FFFDF7"
    SYSTEM_BG = "#EEE2CA"
    SYSTEM_FG = "#6B573D"
    ERROR_BG = "#F5DCD1"
    ERROR_FG = "#873D20"


class Fonts:
    def __init__(self, root):
        try:
            available = {x.lower() for x in tkfont.families(root)}
        except Exception:
            available = set()
        self.serif = next((x for x in ("Georgia", "Book Antiqua", "Palatino Linotype", "DejaVu Serif")
                           if x.lower() in available), "Times")
        self.sans = next((x for x in ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans", "Arial")
                          if x.lower() in available), "Helvetica")


def hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_hex(v):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(x))) for x in v))


def blend(a, b, amount):
    x, y = hex_rgb(a), hex_rgb(b)
    return rgb_hex([x[i] + (y[i] - x[i]) * amount for i in range(3)])


def rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    r = min(r, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
    points = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
              x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
              x1,y2, x1,y2-r, x1,y1+r, x1,y1]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class Motifs:
    @staticmethod
    def mandala(canvas, cx, cy, radius, color):
        for r in (radius * .38, radius * .68, radius):
            canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=color, width=1)
        for i in range(16):
            a = i * math.pi / 8
            x = cx + radius * math.cos(a)
            y = cy + radius * math.sin(a)
            canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline="")

    @staticmethod
    def dots(canvas, w, h, color):
        for y in range(25, h, 58):
            for x in range(25, w, 58):
                canvas.create_oval(x-1, y-1, x+1, y+1, fill=color, outline="")


class RoundedButton(tk.Canvas):
    PALETTE = {
        "primary": (Theme.TERRACOTTA, "#C7653D", Theme.TERRACOTTA_DARK, Theme.CREAM),
        "accent": (Theme.ORANGE, "#EA8842", "#B85A1D", Theme.CREAM),
        "gold": (Theme.GOLD, "#D6A441", "#A97820", Theme.BROWN),
        "quiet": ("#E9DEC9", "#E0D0B5", "#D0B995", Theme.BROWN),
        "dark": (Theme.BROWN, "#51341C", "#241609", Theme.CREAM),
        "ghost": (Theme.PANEL, Theme.BG, Theme.LINE, Theme.BROWN),
    }

    def __init__(self, master, app, text, command, kind="primary", width=150, height=46):
        super().__init__(master, width=width, height=height, bg=Theme.PANEL,
                         highlightthickness=0, bd=0, takefocus=True)
        self.app, self.text, self.command = app, text, command
        self.kind, self.enabled, self.hover, self.pressed = kind, True, False, False
        self.w, self.h = width, height
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Return>", lambda e: self.invoke())
        self.bind("<space>", lambda e: self.invoke())
        self.bind("<Configure>", lambda e: self._configure(e))
        self.draw()

    def _set_hover(self, value):
        self.hover = value and self.enabled
        self.configure(cursor="hand2" if self.hover else "")
        self.draw()

    def _press(self, _):
        if self.enabled:
            self.pressed = True
            self.focus_set()
            self.draw()

    def _release(self, _):
        if self.enabled:
            was = self.pressed
            self.pressed = False
            self.draw()
            if was:
                self.invoke()

    def _configure(self, event):
        self.w, self.h = event.width, event.height
        self.draw()

    def set_enabled(self, value):
        self.enabled = bool(value)
        self.pressed = False
        self.draw()

    def set_text(self, value):
        self.text = value
        self.draw()

    def invoke(self):
        if self.enabled and self.command:
            self.command()

    def draw(self):
        self.delete("all")
        normal, hover, pressed, fg = self.PALETTE.get(self.kind, self.PALETTE["primary"])
        if not self.enabled:
            body, text = blend(normal, Theme.BG, .65), Theme.MUTED
        elif self.pressed:
            body, text = pressed, fg
        elif self.hover:
            body, text = hover, fg
        else:
            body, text = normal, fg
        rounded_rect(self, 2, 4, self.w-1, self.h-1, 13, fill=Theme.SHADOW, outline="")
        rounded_rect(self, 1, 1, self.w-2, self.h-3, 13, fill=body, outline=body)
        self.create_text(self.w/2, self.h/2-1, text=self.text, fill=text,
                         font=(self.app.fonts.sans, 10, "bold"))


class StateStone(tk.Canvas):
    COLORS = {"idle": Theme.MUTED, "listening": Theme.GREEN,
              "thinking": Theme.GOLD, "speaking": Theme.ORANGE,
              "error": Theme.TERRACOTTA_DARK}
    LABELS = {"idle": "READY", "listening": "LISTENING", "thinking": "THINKING",
              "speaking": "SPEAKING", "error": "ATTENTION"}

    def __init__(self, master, app, size=165):
        super().__init__(master, width=size, height=size+38, bg=Theme.PANEL,
                         highlightthickness=0, bd=0)
        self.app, self.size, self.state, self.phase = app, size, "idle", 0
        self.bind("<Configure>", lambda e: self.draw())
        self.draw()

    def set_state(self, state):
        self.state = state if state in self.COLORS else "idle"
        self.phase = 0
        self.draw()

    def tick(self, dt):
        self.phase += dt
        self.draw()

    def draw(self):
        self.delete("all")
        w = max(self.size, self.winfo_width())
        cx, cy, r = w/2, self.size/2, self.size*.40
        color = self.COLORS[self.state]
        self.create_oval(cx-r-5, cy-r-5, cx+r+5, cy+r+5, fill=Theme.BG, outline="")
        self.create_oval(cx-r, cy-r, cx+r, cy+r, fill=blend(color, Theme.CREAM, .88),
                         outline=blend(color, Theme.CREAM, .45), width=2)
        if self.state == "idle":
            Motifs.mandala(self, cx, cy, self.size*.25, blend(color, Theme.CREAM, .2))
        elif self.state == "listening":
            for i in range(3):
                p = (self.phase*.75 + i/3) % 1
                rr = self.size*.08 + self.size*.30*p
                self.create_oval(cx-rr, cy-rr, cx+rr, cy+rr, outline=blend(Theme.CREAM, color, 1-p), width=2)
            self.create_oval(cx-9, cy-9, cx+9, cy+9, fill=color, outline="")
        elif self.state == "thinking":
            for i in range(8):
                a = self.phase*2 + i*math.pi/4
                rr = self.size*.28
                x, y = cx+rr*math.cos(a), cy+rr*math.sin(a)
                pulse = .5 + .5*math.sin(self.phase*4-i*.6)
                d = 3 + pulse*3
                self.create_oval(x-d,y-d,x+d,y+d,fill=blend(Theme.CREAM,color,.35+.65*pulse),outline="")
        elif self.state == "speaking":
            for i in range(9):
                x = cx-self.size*.24+i*self.size*.48/8
                amp = self.size*.045+self.size*.14*abs(math.sin(self.phase*5+i*.7))
                rounded_rect(self,x-3,cy-amp,x+3,cy+amp,3,fill=color,outline="")
        else:
            self.create_text(cx, cy, text="!", fill=color, font=(self.app.fonts.serif, 42, "bold"))
        label = self.LABELS[self.state]
        pw = max(90, len(label)*9+26)
        rounded_rect(self,cx-pw/2,self.size+3,cx+pw/2,self.size+29,13,
                     fill=blend(color,Theme.CREAM,.82),outline=blend(color,Theme.CREAM,.45))
        self.create_text(cx,self.size+16,text=label,fill=blend(color,Theme.BROWN,.25),
                         font=(self.app.fonts.sans,8,"bold"))


class Chat(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=Theme.PANEL, highlightthickness=1, highlightbackground=Theme.LINE)
        self.app = app
        head = tk.Frame(self, bg=Theme.BG, height=44); head.pack(fill="x"); head.pack_propagate(False)
        tk.Label(head,text="CONVERSATION",bg=Theme.BG,fg=Theme.BROWN,
                 font=(app.fonts.sans,9,"bold")).pack(side="left",padx=16)
        self.status = tk.Label(head,text="",bg=Theme.BG,fg=Theme.MUTED,font=(app.fonts.sans,9))
        self.status.pack(side="right",padx=16)
        body=tk.Frame(self,bg=Theme.PANEL); body.pack(fill="both",expand=True)
        self.canvas=tk.Canvas(body,bg=Theme.PANEL,highlightthickness=0,bd=0)
        sb=tk.Scrollbar(body,orient="vertical",command=self.canvas.yview,width=10); sb.pack(side="right",fill="y")
        self.canvas.pack(side="left",fill="both",expand=True); self.canvas.configure(yscrollcommand=sb.set)
        self.inner=tk.Frame(self.canvas,bg=Theme.PANEL)
        self.win=self.canvas.create_window((0,0),window=self.inner,anchor="nw")
        self.inner.bind("<Configure>",lambda e:self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",lambda e:self.canvas.itemconfigure(self.win,width=e.width))
        self.canvas.bind("<MouseWheel>",lambda e:self.canvas.yview_scroll(-int(e.delta/120),"units"))

    def add(self, role, text):
        bg, fg, name, side = {
            "user":(Theme.USER_BG,Theme.USER_FG,"YOU","right"),
            "assistant":(Theme.BOT_BG,Theme.BROWN,"ART EXPERT","left"),
            "error":(Theme.ERROR_BG,Theme.ERROR_FG,"ART EXPERT","left"),
        }.get(role,(Theme.SYSTEM_BG,Theme.SYSTEM_FG,"ART EXPERT","left"))
        row=tk.Frame(self.inner,bg=Theme.PANEL); row.pack(fill="x",padx=8,pady=3)
        bubble=tk.Frame(row,bg=bg,highlightthickness=1,highlightbackground=Theme.LINE)
        bubble.pack(side=side,padx=4)
        tk.Label(bubble,text=name,bg=bg,fg=fg,font=(self.app.fonts.sans,8,"bold"),anchor="w").pack(fill="x",padx=14,pady=(9,1))
        tk.Label(bubble,text=text,bg=bg,fg=fg,font=(self.app.fonts.sans,11),justify="left",anchor="w",
                 wraplength=650).pack(fill="x",padx=14,pady=(1,11))
        self.after(20,lambda:self.canvas.yview_moveto(1))

    def clear(self):
        for child in self.inner.winfo_children(): child.destroy()


class Info(tk.Frame):
    def __init__(self, master, app, label, value="—"):
        super().__init__(master,bg=Theme.PANEL)
        tk.Label(self,text=label.upper(),bg=Theme.PANEL,fg=Theme.MUTED,
                 font=(app.fonts.sans,8,"bold"),anchor="w").pack(fill="x")
        self.value=tk.Label(self,text=value,bg=Theme.PANEL,fg=Theme.BROWN,
                            font=(app.fonts.serif,15,"bold"),anchor="w",justify="left",wraplength=280)
        self.value.pack(fill="x",pady=(2,0))
    def set(self,value): self.value.configure(text=value or "—")

def run_warli():
    subprocess.Popen(["python", "warli.py"])

def run_mandala():
    subprocess.Popen(["python", "mandala.py"])

def run_alpana():
    subprocess.Popen(["python", "alpana.py"])


class ArtExpertPage(tk.Frame):
    MAX_SECONDS = 7.0

    def __init__(self, master, app):
        super().__init__(master,bg=Theme.BG)
        self.app=app
        self.session_active=False
        self.recording=False
        self.record_stop=threading.Event()
        self.session_stop=threading.Event()
        self.record_thread=None
        self.record_done=threading.Event()
        self.record_text=""
        self.record_ok=False
        self.record_error=""
        self.worker=None
        # Explicit command queue: RETRY is a normal loop transition,
        # never an implicit False/stop result.
        self.confirm_queue=queue.Queue()
        self.question_count=0
        self.art_context={}
        self.session=None
        self.state="idle"
        self._build()

    def _build(self):
        top=tk.Frame(self,bg=Theme.BG); top.pack(fill="x",padx=26,pady=(18,10))
        tk.Label(top,text="ART EXPERT",bg=Theme.BG,fg=Theme.TERRACOTTA,font=(self.app.fonts.sans,9,"bold")).pack(side="left")
        tk.Label(top,text="Indian Folk Art Companion",bg=Theme.BG,fg=Theme.BROWN,font=(self.app.fonts.serif,24,"bold")).pack(side="left",padx=16)
        self.clock=tk.Label(top,text="",bg=Theme.BG,fg=Theme.MUTED,font=(self.app.fonts.sans,9)); self.clock.pack(side="right")

        body=tk.Frame(self,bg=Theme.BG); body.pack(fill="both",expand=True,padx=26)
        body.columnconfigure(0,weight=1); body.columnconfigure(1,weight=0,minsize=320); body.rowconfigure(0,weight=1)
        self.chat=Chat(body,self.app); self.chat.grid(row=0,column=0,sticky="nsew",padx=(0,16))

        side=tk.Frame(body,bg=Theme.PANEL,highlightthickness=1,highlightbackground=Theme.LINE)
        side.grid(row=0,column=1,sticky="nsew")
        tk.Label(side,text="STATUS",bg=Theme.BG,fg=Theme.BROWN_LIGHT,font=(self.app.fonts.sans,9,"bold")).pack(fill="x",anchor="w",padx=16,pady=12)
        inner=tk.Frame(side,bg=Theme.PANEL); inner.pack(fill="both",expand=True,padx=18,pady=8)
        self.stone=StateStone(inner,self.app,160); self.stone.pack(pady=(0,14)); self.app.animations.append(self.stone)
        self.art=Info(inner,self.app,"Art form","Not identified yet"); self.art.pack(fill="x",pady=(0,10))
        self.origin=Info(inner,self.app,"Origin"); self.origin.pack(fill="x",pady=(0,10))
        self.conf=Info(inner,self.app,"Confidence"); self.conf.pack(fill="x",pady=(0,10))
        self.questions=Info(inner,self.app,"Questions","0"); self.questions.pack(fill="x",pady=(0,10))
        self.hint=tk.Label(inner,text="Press Start session to begin.",bg=Theme.PANEL,fg=Theme.MUTED,
                           font=(self.app.fonts.sans,10),wraplength=275,justify="left",anchor="w")
        self.hint.pack(fill="x",side="bottom",pady=8)

        controls=tk.Frame(self,bg=Theme.PANEL,highlightthickness=1,highlightbackground=Theme.LINE); controls.pack(fill="x",padx=26,pady=16)
        tk.Frame(controls,bg=Theme.TERRACOTTA,height=5).pack(fill="x")
        row=tk.Frame(controls,bg=Theme.PANEL); row.pack(fill="x",padx=12,pady=12)
        self.start_btn=RoundedButton(row,self.app,"Start session",self.start_session,"primary",150,46); self.start_btn.pack(side="left",padx=(0,8))
        self.record_btn=RoundedButton(row,self.app,"⏺  Record",self.record_or_stop,"accent",150,46); self.record_btn.pack(side="left",padx=(0,8))
        self.confirm_btn=RoundedButton(row,self.app,"✓  Confirm",self.confirm,"gold",130,46); self.confirm_btn.pack(side="left",padx=(0,8))
        self.retry_btn=RoundedButton(row,self.app,"↻  Retry",self.retry,"quiet",120,46); self.retry_btn.pack(side="left",padx=(0,8))
        self.stop_btn=RoundedButton(row,self.app,"Stop",self.stop_session,"dark",95,46); self.stop_btn.pack(side="left")
        RoundedButton(row,self.app,"Home",self.go_home,"ghost",95,46).pack(side="right")
        self._update_controls()

    def set_state(self,state,hint=None):
        self.state=state
        visual={"idle":"idle","ready":"idle","recording":"listening","listening":"listening",
                "thinking":"thinking","speaking":"speaking","confirm":"idle","error":"error"}.get(state,"idle")
        self.stone.set_state(visual); self.chat.status.configure(text=state.upper())
        if hint: self.hint.configure(text=hint)
        self._update_controls()

    def _update_controls(self):
        active=self.session_active
        self.start_btn.set_enabled(not active)
        self.record_btn.set_enabled(active and self.state in {"ready","recording"})
        self.record_btn.set_text("■  Stop recording" if self.recording else "⏺  Record")
        self.confirm_btn.set_enabled(active and self.state=="confirm")
        self.retry_btn.set_enabled(active and self.state=="confirm")
        self.stop_btn.set_enabled(active)

    def handle_key(self,event):
        if not self.session_active: return
        key=(event.keysym or "").lower()
        if key=="return":
            self.record_or_stop()
        elif key=="y": self.confirm()
        elif key in {"r"}: self.retry()
        elif key=="escape": self.stop_session()

    def start_session(self):
        if self.session_active: return
        self.session_active=True; self.session_stop.clear(); self.question_count=0
        self.questions.set("0"); self.chat.clear(); self._started=time.time()
        self.chat.add("system","Starting the Art Expert...")
        self.set_state("thinking","Preparing the camera, speech and AI...")
        self.worker=threading.Thread(target=self._session_worker,daemon=True); self.worker.start()

    def _session_worker(self):
        try:
            backend.setup_logging()
            for required in (backend.BASE_DIR / "prompt.txt", backend.BASE_DIR / "expert_prompt.txt"):
                if not required.exists():
                    raise FileNotFoundError(f"Missing required file: {required}")
            speech.get_vosk_model()
            try:
                tts.init_tts()
            except Exception as exc:
                self.app.post(self.chat.add,"error",f"TTS is unavailable: {exc}")
            if self.session_stop.is_set(): return
            capture = vision.capture_artwork(show_preview=False, auto_retry=True)
            if not capture.get("success"):
                raise RuntimeError(capture.get("reason", "Could not capture the artwork."))
            image_path = capture["image_path"]
            self.set_state("thinking","Looking at the artwork...")
            art_context=vision.identify_art(image_path)
            if not isinstance(art_context,dict): raise RuntimeError("vision.identify_art() did not return a dictionary.")
            self.art_context=art_context
            self.app.post(self.art.set,str(art_context.get("art_style","Unknown")))
            self.app.post(self.origin.set,str(art_context.get("origin","Unknown")))
            c=art_context.get("confidence"); self.app.post(self.conf.set,"—" if c is None else f"{round(float(c)*100 if float(c)<=1 else float(c))}%")
            self.session=backend.build_session(art_context)
            self.speak_show(backend.build_intro_message(art_context))
            self.speak_show("Do you have any questions?")
            self.conversation_loop()
        except Exception as exc:
            logging.exception("UI session failed")
            self.app.post(self.chat.add,"error",f"Something went wrong:\n{exc}")
            self.app.post(self.set_state,"error","The session encountered an error.")
        finally:
            self.app.post(self.finish_session)

    def conversation_loop(self):
        failures=0
        while not self.session_stop.is_set():
            self.app.post(self.set_state,"ready","Press Record or Enter to start speaking.")
            text=self.record_and_transcribe()
            if self.session_stop.is_set(): return
            if not text:
                failures+=1
                if failures>=3:
                    self.app.post(self.chat.add,"error","I'm having trouble hearing right now. Let's pause here.")
                    return
                continue
            failures=0
            self.app.post(self.chat.add,"user",f'I heard: "{text}"')

            # Confirmation has THREE outcomes:
            #   yes   -> continue to Gemini
            #   retry -> go back to recording, same session
            #   stop  -> end the session
            choice = self.wait_confirmation()
            if choice == "retry":
                # IMPORTANT: retry only restarts the recording cycle.
                # The session, Gemini context and artwork context remain alive.
                self.app.post(self.set_state,"ready","Retrying: press Record or Enter to speak again.")
                continue
            if choice == "stop":
                return
            # Only an explicit YES reaches Gemini.
            if choice != "yes":
                continue

            if backend.is_exit_command(text):
                self.speak_show("Okay. Ending the session."); return
            response=self.ask_expert(text)
            if response is None: continue
            if not isinstance(response,dict):
                self.app.post(self.chat.add,"error","The expert returned an invalid response."); continue
            answer=backend.make_spoken_response(response) or str(response.get("summary","")).strip()
            if not answer: answer="I couldn't find a clear answer to that question."
            self.app.post(self.chat.add,"assistant",answer)
            self.question_count+=1; self.app.post(self.questions.set,str(self.question_count))
            backend.append_history(self.session,text,answer)
            self.speak(answer)
            follow=str(response.get("follow_up","")).strip() or "Do you have another question?"
            self.speak_show(follow)

    def record_and_transcribe(self):
        self.record_done.clear()
        self.record_text = ""
        self.record_ok = False
        self.record_error = ""
        self.app.post(self.start_recording)
        while not self.record_done.wait(.1):
            if self.session_stop.is_set():
                return ""
        return self.record_text if self.record_ok else ""

    def record_or_stop(self):
        if not self.session_active: return
        if self.recording:
            self.record_stop.set()
        elif self.state=="ready":
            self.start_recording()

    def start_recording(self):
        self.recording=True; self.record_stop.clear()
        self.set_state("recording","Listening... speak now. Press Enter or Stop recording to finish.")
        self.record_thread=threading.Thread(target=self._record_worker,daemon=True); self.record_thread.start()

    def _record_worker(self):
        path=speech.DEFAULT_AUDIO_PATH
        frames=[]
        start=time.monotonic()
        try:
            def callback(indata, frame_count, time_info, status):
                if status: logging.warning("Audio status: %s",status)
                frames.append(bytes(indata))
            with sd.RawInputStream(samplerate=speech.DEFAULT_SAMPLE_RATE,blocksize=8000,
                                   channels=1,dtype="int16",callback=callback):
                while not self.record_stop.is_set() and not self.session_stop.is_set():
                    if time.monotonic()-start >= self.MAX_SECONDS: break
                    time.sleep(.05)
            if not frames: raise RuntimeError("No audio captured from microphone.")
            path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
            with wave.open(str(path),"wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(speech.DEFAULT_SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
            text=speech.speech_to_text(path).strip()
            self.app.post(self._record_finished,text,True,"")
        except Exception as exc:
            self.app.post(self._record_finished,"",False,str(exc))

    def _record_finished(self,text,ok,error):
        self.recording=False
        self.record_text = text
        self.record_ok = ok
        self.record_error = error
        self.record_done.set()
        if self.session_stop.is_set(): return
        self.set_state("ready" if not ok or not text else "confirm",
                       "Press Record to try again." if not ok or not text else "Confirm what I heard, or Retry.")
        if not ok:
            self.chat.add("error",f"Recording failed: {error}")
            return
        if not text:
            self.chat.add("system","I didn't catch that. Press Record to try again.")

    def wait_confirmation(self):
        # Drain stale commands from a previous confirmation cycle.
        while True:
            try:
                self.confirm_queue.get_nowait()
            except queue.Empty:
                break

        self.app.post(self.set_state,"confirm","Confirm the question, or press Retry to record again.")

        while not self.session_stop.is_set():
            try:
                command=self.confirm_queue.get(timeout=.1)
            except queue.Empty:
                continue

            if command in {"yes","retry","stop"}:
                return command

        return "stop"

    def confirm(self):
        if self.session_active and self.state=="confirm":
            self.chat.add("system","Confirmed.")
            self.confirm_queue.put("yes")

    def retry(self):
        if self.session_active and self.state=="confirm":
            self.chat.add("system","Let's try that again.")
            # Do NOT touch session_stop. Do NOT call finish_session.
            # The worker receives 'retry' and loops back to recording.
            self.confirm_queue.put("retry")

    def ask_expert(self,question):
        self.app.post(self.set_state,"thinking","Gemini is thinking...")
        for attempt in range(3):
            if self.session_stop.is_set(): return None
            try:
                return expert.ask_expert(self.session["art"],question,self.session["history"])
            except Exception as exc:
                if attempt<2 and any(x in str(exc).lower() for x in ("503","500","429","timeout","connection","unavailable","reset","json")):
                    time.sleep(2**attempt); continue
                self.app.post(self.chat.add,"error",f"Gemini error: {exc}"); return None
        return None

    def speak(self,text):
        text=(text or "").strip()
        if not text or self.session_stop.is_set(): return
        self.app.post(self.set_state,"speaking","Speaking...")
        try:
            ok=tts.speak(text,wait=True)
            if ok is False: self.app.post(self.chat.add,"error","TTS returned failure. Check the Piper TTS setup/audio output device.")
        except Exception as exc:
            logging.exception("TTS failed")
            self.app.post(self.chat.add,"error",f"Speech output failed: {exc}")
        finally:
            if not self.session_stop.is_set(): self.app.post(self.set_state,"idle","Ready.")

    def speak_show(self,text):
        self.app.post(self.chat.add,"assistant",text)
        self.speak(text)

    def finish_session(self):
        self.session_active=False; self.recording=False
        self.set_state("idle","Session finished. Press Start session to begin again.")
        self._update_controls()

    def stop_session(self):
        if not self.session_active: return
        self.session_stop.set(); self.record_stop.set(); self.confirm_queue.put("stop"); self.record_done.set()
        self.set_state("idle","Stopping...")

    def go_home(self):
        if self.session_active: self.stop_session()
        self.app.show_page("home")

    def tick(self,_dt):
        if self.session_active and hasattr(self,"_started"):
            e=int(time.time()-self._started); self.clock.configure(text=f"Session {e//60:02d}:{e%60:02d}")


class HomePage(tk.Frame):
    def __init__(self,master,app):
        super().__init__(master,bg=Theme.BG); self.app=app
        canvas=tk.Canvas(self,bg=Theme.BG,highlightthickness=0); canvas.pack(fill="both",expand=True); self.bg=canvas
        canvas.bind("<Configure>",self.draw_bg)
        content=tk.Frame(self,bg=Theme.BG); content.place(relx=0,rely=0,relwidth=1,relheight=1)
        tk.Label(content,text="INDIAN FOLK ART",bg=Theme.BG,fg=Theme.TERRACOTTA,font=(app.fonts.sans,10,"bold")).pack(pady=(45,8))
        tk.Label(content,text="Art Expert",bg=Theme.BG,fg=Theme.BROWN,font=(app.fonts.serif,38,"bold")).pack()
        tk.Label(content,text="Discover the story behind the artwork.",bg=Theme.BG,fg=Theme.MUTED,font=(app.fonts.sans,12)).pack(pady=(6,28))
        card=tk.Frame(content,bg=Theme.PANEL,highlightthickness=1,highlightbackground=Theme.LINE); card.pack(padx=60,pady=10)
        stone=StateStone(card,app,185); stone.pack(padx=45,pady=(24,8))
        tk.Label(card,text="The artwork is already selected.\nStart the expert and ask your questions by voice.",
                 bg=Theme.PANEL,fg=Theme.BROWN,font=(app.fonts.sans,11),justify="center").pack(padx=40,pady=(4,20))
        RoundedButton(card,app,"Start Art Expert",lambda:app.show_page("expert"),"primary",210,52).pack(pady=(0,12))

        # Simple full-program shutdown button.
        # Uses the existing App.quit_app() path so the working
        # recording/retry/session logic remains untouched.
        RoundedButton(
            content,
            app,
            "Quit Program",
            app.quit_app,
            "quiet",
            150,
            40,
        ).pack(side="bottom",pady=(0,8))

        row2 = tk.Frame(content, bg=Theme.PANEL)
        row2.pack(side="bottom", pady=(10, 20))

        RoundedButton(row2, app, "Draw Warli", run_warli, "primary", 150, 40).pack(side="left", padx=6)
        RoundedButton(row2, app, "Draw Mandala", run_mandala, "accent", 150, 40).pack(side="left", padx=6)
        RoundedButton(row2, app, "Draw Alpana", run_alpana, "gold", 150, 40).pack(side="left", padx=6)

        tk.Label(
            content,
            text="Folk Art AI • Interactive Learning",
            bg=Theme.BG,
            fg=Theme.MUTED,
            font=(app.fonts.sans,9),
        ).pack(side="bottom",pady=(0,12))

    def draw_bg(self,event):
        self.bg.delete("all")
        Motifs.dots(self.bg,event.width,event.height,blend(Theme.BG,Theme.GOLD,.15))
        Motifs.mandala(self.bg,90,event.height-90,105,blend(Theme.BG,Theme.TERRACOTTA,.12))
        Motifs.mandala(self.bg,event.width-90,event.height-90,105,blend(Theme.BG,Theme.TERRACOTTA,.12))


class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Indian Folk Art AI Assistant"); self.configure(bg=Theme.BG); self.minsize(1100,700)
        self.geometry("1400x900"); self.fonts=Fonts(self); self.animations=[]; self.queue=[]; self.lock=threading.Lock(); self.fullscreen=False
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.container=tk.Frame(self,bg=Theme.BG); self.container.pack(fill="both",expand=True)
        self.pages={"home":HomePage(self.container,self),"expert":ArtExpertPage(self.container,self)}
        for p in self.pages.values(): p.place(in_=self.container,x=0,y=0,relwidth=1,relheight=1)
        self.current="home"; self.pages["home"].lift()
        self.bind_all("<Key>",self.global_key); self.bind("<F11>",lambda e:self.toggle_fullscreen()); self.protocol("WM_DELETE_WINDOW",self.quit_app)
        self.after(25,self.pump); self.after(33,self.animate)
        threading.Thread(target=self.init_backend,daemon=True).start()

    def init_backend(self):
        try:
            backend.setup_logging(); speech.get_vosk_model()
            try: tts.init_tts()
            except Exception as exc: logging.exception("TTS init failed: %s",exc)
        except Exception as exc:
            logging.exception("Backend initialization failed")
            self.post(self.pages["expert"].chat.add,"error",f"Backend initialization failed:\n{exc}")

    def post(self,fn,*args):
        with self.lock: self.queue.append((fn,args))

    def pump(self):
        with self.lock: jobs=self.queue[:100]; del self.queue[:len(jobs)]
        for fn,args in jobs:
            try: fn(*args)
            except Exception: logging.exception("UI update failed")
        self.after(25,self.pump)

    def animate(self):
        now=time.perf_counter(); dt=min(.1,now-getattr(self,"_last",now)); self._last=now
        for w in list(self.animations):
            try: w.tick(dt)
            except Exception: pass
        page=self.pages.get(self.current)
        if page and hasattr(page,"tick"): page.tick(dt)
        self.after(33,self.animate)

    def show_page(self,name): self.pages[name].lift(); self.current=name
    def global_key(self,event):
        if self.current=="expert": self.pages["expert"].handle_key(event)
    def toggle_fullscreen(self): self.fullscreen=not self.fullscreen; self.attributes("-fullscreen",self.fullscreen)
    def quit_app(self):
        # Stop any active recording/session first.
        page = self.pages.get("expert")
        if page and page.session_active:
            page.stop_session()

        # Release TTS/audio resources.
        try:
            tts.shutdown()
        except Exception:
            pass

        # Close the entire Tkinter application.
        self.after(50, self.destroy)


if __name__ == "__main__":
    App().mainloop()
