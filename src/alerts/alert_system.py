"""
FEPSIP Alert System
Real-time alerts via Email, Telegram, Webhook, and Slack.
"""
from __future__ import annotations
import asyncio
import json
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional
import httpx
from src.utils import logger


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    ticker: Optional[str] = None
    risk_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def format_message(self) -> str:
        lines = [
            f"🚨 FEPSIP Alert [{self.level.value}]",
            f"📋 {self.title}",
            f"📅 {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            self.message,
        ]
        if self.ticker:
            lines.insert(2, f"📊 Ticker: {self.ticker}")
        if self.risk_score is not None:
            lines.insert(-1, f"⚠️ Risk Score: {self.risk_score:.1f}/100")
        return "\n".join(lines)


class AlertDispatcher:
    """
    Routes alerts to configured channels.
    Each channel is independently enabled/disabled via config.
    """

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        webhook_url: str = "",
        email_config: Optional[dict] = None,
    ) -> None:
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.webhook_url = webhook_url
        self.email_config = email_config or {}
        self._alert_history: list[Alert] = []

    async def dispatch(self, alert: Alert) -> dict[str, bool]:
        """Dispatch alert to all configured channels."""
        self._alert_history.append(alert)
        results = {}

        tasks = []
        if self.telegram_token and self.telegram_chat_id:
            tasks.append(("telegram", self._send_telegram(alert)))
        if self.webhook_url:
            tasks.append(("webhook", self._send_webhook(alert)))
        if self.email_config.get("enabled"):
            tasks.append(("email", self._send_email(alert)))

        for name, coro in tasks:
            try:
                await coro
                results[name] = True
                logger.info("Alert dispatched via {}: {}", name, alert.title)
            except Exception as e:
                results[name] = False
                logger.error("Alert dispatch failed via {}: {}", name, e)

        return results

    async def _send_telegram(self, alert: Alert) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": alert.format_message(),
                "parse_mode": "HTML",
            })

    async def _send_webhook(self, alert: Alert) -> None:
        payload = {
            "level": alert.level.value,
            "title": alert.title,
            "message": alert.message,
            "ticker": alert.ticker,
            "risk_score": alert.risk_score,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(self.webhook_url, json=payload)

    async def _send_email(self, alert: Alert) -> None:
        cfg = self.email_config
        msg = MIMEText(alert.format_message())
        msg["Subject"] = f"[FEPSIP {alert.level.value}] {alert.title}"
        msg["From"] = cfg.get("sender", "")
        msg["To"] = cfg.get("recipient", cfg.get("sender", ""))

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._smtp_send, msg, cfg)

    @staticmethod
    def _smtp_send(msg: MIMEText, cfg: dict) -> None:
        with smtplib.SMTP(cfg.get("smtp_host", ""), cfg.get("smtp_port", 587)) as s:
            s.starttls()
            s.login(cfg.get("sender", ""), cfg.get("password", ""))
            s.send_message(msg)

    @property
    def history(self) -> list[Alert]:
        return list(self._alert_history)


class AlertRuleEngine:
    """Evaluates metrics against alert thresholds and fires alerts."""

    def __init__(
        self,
        dispatcher: AlertDispatcher,
        risk_threshold: float = 80.0,
        contagion_threshold: float = 0.7,
    ) -> None:
        self.dispatcher = dispatcher
        self.risk_threshold = risk_threshold
        self.contagion_threshold = contagion_threshold

    async def check_risk_score(self, ticker: str, score: float) -> Optional[Alert]:
        if score >= self.risk_threshold:
            level = AlertLevel.CRITICAL if score >= 90 else AlertLevel.WARNING
            alert = Alert(
                level=level,
                title=f"High Systemic Risk: {ticker}",
                message=f"{ticker} systemic risk score has reached {score:.1f}/100, "
                        f"exceeding the threshold of {self.risk_threshold}. "
                        f"Consider reducing exposure or hedging.",
                ticker=ticker,
                risk_score=score,
            )
            await self.dispatcher.dispatch(alert)
            return alert
        return None

    async def check_anomaly(self, ticker: str, anomaly_type: str, severity: float) -> Optional[Alert]:
        if severity >= 0.6:
            alert = Alert(
                level=AlertLevel.CRITICAL if severity >= 0.9 else AlertLevel.WARNING,
                title=f"Anomaly Detected: {ticker} – {anomaly_type}",
                message=f"An anomaly of type '{anomaly_type}' was detected for {ticker} "
                        f"with severity {severity:.2f}. This may indicate unusual market activity.",
                ticker=ticker,
                metadata={"anomaly_type": anomaly_type, "severity": severity},
            )
            await self.dispatcher.dispatch(alert)
            return alert
        return None

    async def check_contagion(self, source: str, spread: float, affected_count: int) -> Optional[Alert]:
        if spread >= self.contagion_threshold:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                title=f"Contagion Event: {source}",
                message=f"Contagion spreading from {source} with intensity {spread:.2f}. "
                        f"{affected_count} entities are at risk of impact.",
                ticker=source,
                metadata={"spread": spread, "affected_count": affected_count},
            )
            await self.dispatcher.dispatch(alert)
            return alert
        return None
