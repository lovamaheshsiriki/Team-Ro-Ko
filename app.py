"""
Job/Internship Mail Matcher & Prep
- Gmail IMAP: Fetch emails, extract only job/internship opportunities (structured)
- Filter: Skip newsletters, articles, courses; extract company, role, stipend, deadline, location, applyLink
- Resume upload & skill extraction
- Shortlist opportunities by resume skill match
- Match analysis: gap, probability, verdict
- Preparation tips per opportunity
"""

import streamlit as st
import os
import io
import re
import json
import imaplib
import email

import cv2
import numpy as np
from email.header import decode_header
from groq import Groq
from pypdf import PdfReader
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()

client = Groq(api_key=api_key)

# --- CONFIG ---
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# --- UTILITIES ---

def text_to_speech(text: str):
    """Convert text to speech and return MP3 bytes."""
    try:
        tts = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe spoken answer using Groq Whisper."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "answer.wav"
        result = client.audio.transcriptions.create(
            file=("answer.wav", audio_file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
        return result
    except Exception:
        return ""


def analyze_face_focus(image_bytes: bytes) -> str:
    """Very simple face/eye-focus heuristic using OpenCV."""
    try:
        file_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if frame is None:
            return "Unable to analyze face."
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return "Face not clearly visible – please sit in front of the camera."

        h, w = gray.shape
        msg_parts = []
        for (x, y, fw, fh) in faces[:1]:  # first detected face
            cx = x + fw / 2
            cy = y + fh / 2
            # Head position vs frame center
            dx = abs(cx - w / 2) / (w / 2)
            dy = abs(cy - h / 2) / (h / 2)
            if dx < 0.3 and dy < 0.3:
                msg_parts.append("Head position: centered (good).")
            else:
                msg_parts.append("Head position: off-center – try to sit in the middle of the frame.")

            roi_gray = gray[y : y + fh, x : x + fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            if len(eyes) >= 2:
                msg_parts.append("Eye contact: looks good – eyes detected clearly.")
            elif len(eyes) == 1:
                msg_parts.append("Eye contact: partially detected – ensure you look at the camera.")
            else:
                msg_parts.append("Eye contact: not clear – raise your head and look at the camera.")
        return " ".join(msg_parts)
    except Exception:
        return "Unable to analyze face."


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "".join([p.extract_text() for p in reader.pages if p.extract_text()])


def decode_mime_header(s):
    if s is None:
        return ""
    decoded = decode_header(s)
    parts = []
    for part, enc in decoded:
        if isinstance(part, bytes):
            parts.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(str(part))
    return " ".join(parts)


def get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        break
                except Exception:
                    pass
            elif ctype == "text/html":
                try:
                    payload = part.get_payload(decode=True)
                    if payload and not body:
                        raw = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        body = re.sub(r"<[^>]+>", " ", raw)
                        body = re.sub(r"\s+", " ", body).strip()
                        break
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            pass
    return body[:3000]  # Limit for API


def fetch_emails_imap(gmail_user: str, app_password: str, folder: str = "INBOX", max_emails: int = 50):
    """Fetch emails from Gmail via IMAP."""
    emails_list = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(gmail_user, app_password)
        mail.select(folder)
        _, data = mail.search(None, "ALL")
        msg_ids = data[0].split()
        msg_ids = msg_ids[-max_emails:] if len(msg_ids) > max_emails else msg_ids

        for mid in reversed(msg_ids):
            try:
                _, msg_data = mail.fetch(mid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                subject = decode_mime_header(msg.get("Subject", ""))
                from_addr = decode_mime_header(msg.get("From", ""))
                body = get_email_body(msg)
                emails_list.append({
                    "id": mid.decode() if isinstance(mid, bytes) else mid,
                    "subject": subject,
                    "from": from_addr,
                    "body": body,
                    "raw_msg": msg,
                })
            except Exception as e:
                continue
        mail.logout()
    except Exception as e:
        raise RuntimeError(f"IMAP error: {str(e)}")
    return emails_list


def extract_job_opportunity(subject: str, body: str, from_addr: str) -> dict | None:
    """
    Extract structured job/internship data from email ONLY if it clearly contains an opportunity.
    Returns None for newsletters, articles, updates, skill courses, etc.
    """
    prompt = f"""You are an AI system that extracts internship or job opportunities from email and Telegram messages.

Your task is to analyze this message and return structured opportunity data ONLY when the message clearly describes a job or internship opportunity.

STRICT RULES:
1. Only extract messages that contain a real job or internship opportunity.
2. If the message is a newsletter, article, update, skill course, personal notification, or unrelated content, return {{"skip": true, "reason": "brief reason"}}.
3. Never return an opportunity where company OR role is unknown.
4. Normalize company names if possible.

Extract these fields when it IS an opportunity:
- company, role, stipend, deadline, location, applyLink

DEADLINE: Extract only the date, remove time. E.g. "14 March 2026 (11:59 PM)" → "14 March 2026"
LINK: Extract full URL or link. Use "Unknown" if none found.

SKIP: newsletters, career articles, skill courses, personal emails, messages without job roles.

Message:
From: {from_addr}
Subject: {subject}
Body: {body[:2500]}

Return ONLY a valid JSON object. If opportunity: {{"company":"X","role":"Y","stipend":"...","deadline":"...","location":"...","applyLink":"..."}}
If NOT an opportunity: {{"skip": true, "reason": "..."}}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = res.choices[0].message.content.strip()
        start = text.find("{")
        if start >= 0:
            depth, end = 0, start
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                data = json.loads(text[start:end])
                if data.get("skip"):
                    return None
                company = data.get("company", "").strip()
                role = data.get("role", "").strip()
                if not company or not role:
                    return None
                return {
                    "company": company,
                    "role": role,
                    "stipend": str(data.get("stipend", "Unknown")),
                    "deadline": str(data.get("deadline", "Unknown")),
                    "location": str(data.get("location", "Unknown")),
                    "applyLink": str(data.get("applyLink", "Unknown")),
                }
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        pass
    return None


def extract_skills_from_resume(resume_text: str) -> list:
    """Extract skills from resume using LLM."""
    if not resume_text or not resume_text.strip():
        return []

    prompt = f"""List all technical and soft skills from this resume. Reply with ONLY a valid JSON array of strings. No explanation, no markdown, no code blocks. Example format: ["Python", "Machine Learning", "Communication", "Teamwork"]

Resume text:
{resume_text[:5000]}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = res.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.strip()

        # Try to parse JSON array (handle nested brackets)
        start = text.find("[")
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "[":
                    depth += 1
                elif c == "]":
                    depth -= 1
                    if depth == 0:
                        arr = json.loads(text[start : i + 1])
                        if isinstance(arr, list):
                            return [str(s).strip() for s in arr if s]
                        break

        # Fallback: extract quoted strings like "Python" or 'Java'
        quoted = re.findall(r'"([^"]+)"', text) or re.findall(r"'([^']+)'", text)
        if quoted:
            skills = [s.strip() for s in quoted if 2 < len(s.strip()) < 50]
            if skills:
                return skills[:25]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return []


def compute_quick_match_score(resume_skills: list, required_skills: list) -> float:
    """Quick match % for shortlisting."""
    r_set = {s.strip().lower() for s in required_skills if s}
    u_set = {s.strip().lower() for s in resume_skills if s}
    if not r_set:
        return 0.0
    matched = sum(1 for s in r_set if any(s in u or u in s for u in u_set))
    return round(100 * matched / len(r_set), 1)


def extract_required_skills_from_email(subject: str, body: str, role: str = "") -> list:
    """Extract required skills from job/internship email."""
    role_ctx = f"Role: {role}\n" if role else ""
    prompt = f"""Extract required skills, technologies, and qualifications from this job/internship email. Return only a JSON array of skill names. Example: ["Python", "SQL", "React"]

{role_ctx}Subject: {subject}
Body: {body[:2500]}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = res.choices[0].message.content.strip()
        match = re.search(r'\[[^\]]*\]', text)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


def compute_skill_match(resume_skills: list, required_skills: list) -> dict:
    """Compute match %, gaps, and probability."""
    r_set = {s.strip().lower() for s in required_skills if s}
    u_set = {s.strip().lower() for s in resume_skills if s}

    matched = [s for s in r_set if any(s in u or u in s for u in u_set)]
    gaps = list(r_set - set(matched))

    match_pct = round(100 * len(matched) / len(r_set), 1) if r_set else 0
    # Probability: match + bonus for having extra relevant skills
    prob = min(100, match_pct + (5 if len(u_set) > len(r_set) else 0))

    prompt = f"""Given:
- Resume skills: {json.dumps(list(u_set))}
- Required skills: {json.dumps(list(r_set))}
- Matched: {json.dumps(list(matched))}
- Gap (missing): {json.dumps(list(gaps))}
- Match %: {match_pct}%

Provide:
1. Match probability (0-100) for this job/internship
2. Is this a good fit? (YES/MAYBE/NO)
3. Top 3 skills to learn to close the gap
4. One-line verdict
Reply in JSON: {{"probability": N, "fit": "YES|MAYBE|NO", "learn": ["s1","s2","s3"], "verdict": "..."}}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = res.choices[0].message.content.strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "matched": list(matched),
                "gaps": gaps,
                "match_pct": match_pct,
                "probability": int(data.get("probability", match_pct)),
                "fit": str(data.get("fit", "MAYBE")),
                "learn": data.get("learn", list(gaps)[:3]),
                "verdict": str(data.get("verdict", "")),
            }
    except Exception:
        pass
    return {
        "matched": list(matched),
        "gaps": list(gaps),
        "match_pct": match_pct,
        "probability": match_pct,
        "fit": "MAYBE",
        "learn": list(gaps)[:3],
        "verdict": f"Match: {match_pct}%. Gap skills: {', '.join(list(gaps)[:5])}",
    }


def generate_interview_questions(
    company: str, role: str, required_skills: list, resume_text: str, num_questions: int = 5
) -> list:
    """Generate interview questions tailored to the job/internship role."""
    prompt = f"""You are an expert interviewer for {company}. Generate {num_questions} interview questions for the role: {role}.

Required skills: {required_skills}

Consider the candidate's resume (for context):
{resume_text[:1500]}

Generate mix of: technical questions, behavioral (STAR format), and role-specific questions. Return ONLY a JSON array of question strings. Example: ["Question 1?", "Question 2?"]
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = res.choices[0].message.content.strip()
        match = re.search(r'\[[\s\S]*?\]', text)
        if match:
            qs = json.loads(match.group())
            return [str(q).strip() for q in qs if q][:num_questions]
    except Exception:
        pass
    return []


def analyze_answer(question: str, answer: str) -> str:
    """Evaluate answer using STAR method, provide ideal answer and suggestions."""
    system_prompt = (
        "You are an expert Senior Recruiter. Evaluate candidate answers using the STAR method (Situation, Task, Action, Result). "
        "BE FAIR: If the answer is relevant and clear, score 6-9. Exceptionally detailed: 10. Only give <5 if irrelevant or too brief. "
        "Provide a concise ideal/correct answer and 2-3 actionable suggestions for improvement."
    )
    user_prompt = f"""Question: {question}
Candidate Answer: {answer}

Provide the following in this exact format:

### 🏆 Evaluation Score: [X/10]

**📋 Review:**
- **S/T (Situation/Task):** How well was the context set?
- **A (Action):** Were the actions clear and impactful?
- **R (Result):** Was there a measurable outcome?
- **Strengths:** [What the candidate did well]
- **Gaps:** [What was missing or could improve]

**✅ Ideal / Correct Answer:**
[Write 2-4 sentences showing what an excellent answer would look like for this question]

**💡 Suggestions for this question:**
1. [Specific suggestion]
2. [Specific suggestion]
3. [Specific suggestion if needed]
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return res.choices[0].message.content
    except Exception:
        return "Evaluation unavailable."


def get_preparation_tips(
    company: str, role: str, resume_skills: list, required_skills: list, gaps: list, role_type: str = "internship"
) -> str:
    """Generate preparation tips for the job/internship."""
    prompt = f"""You are a career coach. Prepare a candidate for this opportunity:

Company: {company}
Role: {role}

Their skills: {resume_skills}
Required skills: {required_skills}
Missing skills (gap): {gaps}
Role type: {role_type}

Provide 5 specific, actionable preparation tips to succeed in this role. Use markdown bullets.
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return res.choices[0].message.content
    except Exception:
        return "Unable to generate tips. Focus on learning the gap skills."


# --- SESSION STATE ---
def init_session():
    defaults = {
        "emails": [], "opportunities": [], "shortlisted": [], "resume_text": "", "resume_skills": [],
        "gmail_connected": False,
        "interview_questions": [], "interview_current_q": 0, "interview_feedback": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# --- UI ---
st.set_page_config(page_title="Job/Internship Matcher", layout="wide", page_icon="📬")
init_session()

st.title("📬 Job & Internship Mail Matcher")
st.caption("Connect Gmail → Extract Jobs/Internships → Shortlist → Match → Prepare")

t1, t2, t3, t4 = st.tabs(["📥 Gmail & Extract", "📄 Resume & Skills", "🎯 Shortlist & Match", "📚 Prepare"])

# --- TAB 1: Gmail & Extract ---
with t1:
    st.subheader("Connect Gmail (IMAP)")
    st.info("Enable IMAP in Gmail settings. Use an [App Password](https://myaccount.google.com/apppasswords) if 2FA is on.")
    gmail_user = st.text_input("Gmail address", placeholder="you@gmail.com", key="gmail_user")
    app_password = st.text_input("App Password", type="password", placeholder="16-char app password", key="app_pass")

    if st.button("Fetch Emails", key="fetch_btn"):
        if not gmail_user or not app_password:
            st.warning("Enter Gmail and App Password")
        else:
            with st.spinner("Fetching emails..."):
                try:
                    all_emails = fetch_emails_imap(gmail_user, app_password, max_emails=10)
                    st.session_state.emails = all_emails
                    st.session_state.gmail_connected = True
                    st.success(f"Fetched {len(all_emails)} emails")
                except Exception as e:
                    st.error(str(e))

    if st.session_state.emails:
        st.subheader("Extract Job/Internship Opportunities")
        st.caption("Only emails with clear job/internship roles will be extracted. Newsletters, articles, courses are skipped.")
        if st.button("Extract Opportunities", key="extract_opp_btn"):
            opportunities = []
            with st.spinner("Extracting job/internship opportunities..."):
                for e in st.session_state.emails:
                    opp = extract_job_opportunity(e["subject"], e["body"], e["from"])
                    if opp:
                        opp["subject"] = e["subject"]
                        opp["from"] = e["from"]
                        opp["body"] = e["body"]
                        opportunities.append(opp)
            st.session_state.opportunities = opportunities
            st.success(f"Found {len(opportunities)} job/internship opportunities")

        if st.session_state.opportunities:
            st.subheader("Extracted Opportunities")
            for i, o in enumerate(st.session_state.opportunities):
                with st.expander(f"**{o['company']}** | {o['role']}"):
                    st.write(f"**Stipend:** {o['stipend']} | **Deadline:** {o['deadline']} | **Location:** {o['location']}")
                    st.write(f"**Apply:** {o['applyLink']}")
                    st.caption(f"From: {o['from']} | {o['subject'][:80]}...")

# --- TAB 2: Resume & Skills ---
with t2:
    st.subheader("Upload Resume (PDF)")
    uploaded = st.file_uploader("Resume PDF", type=["pdf"], key="resume_upload")
    if uploaded:
        text = extract_text_from_pdf(uploaded)
        st.session_state.resume_text = text
        if not text or len(text.strip()) < 50:
            st.warning("Could not extract enough text from the PDF. Try a text-based PDF (not image-only).")
        else:
            if st.button("Extract Skills", key="extract_skills"):
                with st.spinner("Extracting skills..."):
                    skills = extract_skills_from_resume(text)
                    st.session_state.resume_skills = skills
                    if skills:
                        st.success(f"Found {len(skills)} skills")
                    else:
                        st.warning("No skills extracted. The PDF may be image-based or the format may not be recognized.")
        if st.session_state.resume_skills:
            st.write("**Your skills:**", ", ".join(st.session_state.resume_skills))

# --- TAB 3: Shortlist & Match ---
with t3:
    st.subheader("Shortlist & Match Analysis")
    opps = st.session_state.opportunities
    resume_skills = st.session_state.resume_skills or []

    if not resume_skills:
        st.warning("Upload resume and extract skills first (Resume & Skills tab)")
    elif not opps:
        st.warning("Fetch emails and extract opportunities first (Gmail & Extract tab)")
    else:
        if st.button("Shortlist by Resume Match", key="shortlist_btn"):
            shortlisted = []
            with st.spinner("Computing match scores..."):
                for o in opps:
                    req = extract_required_skills_from_email(o["subject"], o["body"], o.get("role", ""))
                    score = compute_quick_match_score(resume_skills, req)
                    shortlisted.append({
                        **o,
                        "required_skills": req,
                        "match_score": score,
                    })
                shortlisted.sort(key=lambda x: x["match_score"], reverse=True)
                st.session_state.shortlisted = shortlisted
            st.success(f"Shortlisted {len(shortlisted)} opportunities by skill match")

        shortlisted = st.session_state.shortlisted or []
        if shortlisted:
            st.subheader("Shortlisted Opportunities (by match %)")
            selected_idx = st.selectbox(
                "Select an opportunity for full analysis",
                range(len(shortlisted)),
                format_func=lambda i: f"{shortlisted[i]['company']} | {shortlisted[i]['role']} ({shortlisted[i]['match_score']}% match)",
                key="shortlist_select"
            )
            if selected_idx is not None and st.button("Full Match Analysis", key="analyze"):
                o = shortlisted[selected_idx]
                req_skills = o.get("required_skills") or extract_required_skills_from_email(
                    o["subject"], o["body"], o.get("role", "")
                )
                result = compute_skill_match(resume_skills, req_skills)

                st.write(f"**{o['company']}** — {o['role']} | Stipend: {o['stipend']} | Deadline: {o['deadline']}")
                if o.get("applyLink") and o["applyLink"] != "Unknown":
                    link = o["applyLink"]
                    if not link.startswith(("http://", "https://")):
                        link = "https://" + link
                    st.markdown(f"**Apply:** [{o['applyLink']}]({link})")

                c1, c2, c3 = st.columns(3)
                c1.metric("Match %", f"{result['match_pct']}%")
                c2.metric("Fit Probability", f"{result['probability']}%")
                c3.metric("Verdict", result["fit"])

                st.write("**Matched skills:**", ", ".join(result["matched"]) or "None")
                st.write("**Gap (missing):**", ", ".join(result["gaps"]) or "None")
                st.write("**Skills to learn:**", ", ".join(result["learn"]))
                st.info(result["verdict"])

                opp_key = f"{o.get('company','')}_{o.get('role','')}"
                st.session_state["last_match"] = {
                    "result": result,
                    "resume_skills": resume_skills,
                    "required_skills": req_skills,
                    "opportunity": o,
                }
                if st.session_state.get("last_opp_key") != opp_key:
                    st.session_state.last_opp_key = opp_key
                    st.session_state.interview_questions = []
                    st.session_state.interview_current_q = 0
                    st.session_state.interview_feedback = []

# --- TAB 4: Prepare ---
with t4:
    st.subheader("Interview Preparation")
    last = st.session_state.get("last_match")
    if not last:
        st.info("Run a full match analysis first (Shortlist & Match tab)")
    else:
        opp = last.get("opportunity", {})
        st.write(f"**Preparing for:** {opp.get('company', '')} — {opp.get('role', '')}")

        # --- Interview Practice ---
        st.markdown("### 🎙️ Mock Interview Questions")
        questions = st.session_state.interview_questions
        curr_q = st.session_state.interview_current_q
        feedback = st.session_state.interview_feedback

        if not questions:
            if st.button("Generate Interview Questions", key="gen_questions"):
                with st.spinner("Generating questions for this role..."):
                    resume_text = st.session_state.resume_text or ""
                    qs = generate_interview_questions(
                        opp.get("company", ""),
                        opp.get("role", ""),
                        last["required_skills"],
                        resume_text,
                        num_questions=5,
                    )
                    st.session_state.interview_questions = qs
                    st.session_state.interview_current_q = 0
                    st.session_state.interview_feedback = []
                    st.rerun()

        if questions:
            idx = curr_q
            # Show feedback for already-answered questions
            if feedback:
                with st.expander("📝 Past Q&A & Feedback", expanded=False):
                    for i, f in enumerate(feedback):
                        st.markdown(f"**Q{i+1}:** {f['q'][:60]}...")
                        st.markdown(f["feedback"])
                        st.divider()

            if idx < len(questions):
                curr_question = questions[idx]
                answered_this = len(feedback) > idx

                if answered_this:
                    st.success(f"**Question {idx + 1}:** {curr_question}")
                    st.markdown("---")
                    st.markdown("**📋 Your feedback:**")
                    st.markdown(feedback[idx]["feedback"])
                    if st.button("Next Question →", key=f"next_{idx}"):
                        st.session_state.interview_current_q = idx + 1
                        st.rerun()
                else:
                    st.info(f"**Question {idx + 1} of {len(questions)}:** {curr_question}")

                    q_audio = text_to_speech(curr_question)
                    if q_audio:
                        st.audio(q_audio, format="audio/mp3")

                    col_cam, col_ans = st.columns([1, 2])
                    with col_cam:
                        cam_img = st.camera_input("Eye contact & head position", key=f"cam_{idx}")
                        if cam_img is not None:
                            face_msg = analyze_face_focus(cam_img.getvalue())
                            st.caption(face_msg)

                    with col_ans:
                        st.markdown("**Record your answer**")
                        audio = mic_recorder(
                            start_prompt="Start answer",
                            stop_prompt="Stop",
                            just_once=True,
                            key=f"mic_{idx}",
                        )
                        if audio and audio.get("bytes"):
                            with st.spinner("Transcribing answer..."):
                                transcript = transcribe_audio(audio["bytes"])
                            st.text_area(
                                "Transcribed answer (you can edit before submit)",
                                value=transcript,
                                key=f"ans_{idx}",
                                height=140,
                            )
                            if st.button("Submit & Get Feedback", key=f"submit_{idx}"):
                                clean_answer = st.session_state.get(f"ans_{idx}", "").strip()
                                if clean_answer:
                                    with st.spinner("Evaluating with STAR method..."):
                                        eval_text = analyze_answer(curr_question, clean_answer)
                                        feedback.append({"q": curr_question, "a": clean_answer, "feedback": eval_text})
                                        st.session_state.interview_feedback = feedback
                                    st.rerun()
            else:
                st.success("You've completed all questions!")
                for i, f in enumerate(feedback):
                    with st.expander(f"Q{i+1}: {f['q'][:50]}..."):
                        st.markdown(f["feedback"])
                if st.button("🔄 Restart Interview", key="restart_int"):
                    st.session_state.interview_questions = []
                    st.session_state.interview_current_q = 0
                    st.session_state.interview_feedback = []
                    st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Quick Prep Tips")
        role_type = "internship" if "intern" in opp.get("role", "").lower() else "job"
        if st.button("Generate Prep Tips", key="prep_btn"):
            tips = get_preparation_tips(
                opp.get("company", ""),
                opp.get("role", ""),
                last["resume_skills"],
                last["required_skills"],
                list(last["result"]["gaps"]),
                role_type,
            )
            st.markdown(tips)
