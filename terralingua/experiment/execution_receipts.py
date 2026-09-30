"""Host receipts for selected actions, correlated results, and observed effects.

Receipts add evidence to existing history. They never rewrite an agent's memory
or certify its interpretation of a result.
"""

import copy
import json
import uuid

from terralingua.voting.election import VoteOutcome
from terralingua.voting.manager import VotingManager


def request_key(server, agent, request_id):
    return json.dumps([server, agent, request_id])


def reward_contract(manager):
    """Identify the configured strategy without copying its mutable history."""
    if manager is None:
        return None
    rewards = manager.rewards
    return {
        "manager": f"{type(manager).__module__}.{type(manager).__qualname__}",
        "strategy": f"{type(rewards).__module__}.{type(rewards).__qualname__}",
        "parameters": {key: getattr(rewards, key) for key in ("cost", "coefficient")
                       if hasattr(rewards, key)},
    }


class ExecutionReceiptsMixin:
    def _reward_state_checkpoint(self):
        return {server: {"contract": reward_contract(manager), "state": manager.rewards.get_state_ckpt()}
                for server, manager in getattr(self, "voting_managers", {}).items()}

    def _restore_reward_state(self, states):
        for server, saved in states.items():
            manager = getattr(self, "voting_managers", {}).get(server)
            if manager is not None and saved.get("contract") == reward_contract(manager):
                manager.rewards.set_state_ckpt(saved.get("state", {}))

    def _capture_action_requests(self, actions, ts):
        self._action_requests = {}
        for tag, action in actions.items():
            request = {
                "selection_id": uuid.uuid4().hex,
                "step": ts,
                "action": action.get("action"),
                "params": copy.deepcopy(action.get("params") or {}),
                "context": copy.deepcopy(getattr(getattr(self, "env", None), "execution_context", {})),
                "selection_source": action.get("source", "agent"),
                "default_reason": action.get("default_reason"),
            }
            self._action_requests[tag] = request
            agent = self.agents.get(tag)
            if (agent is not None and getattr(agent, "history", None)
                    and action.get("source") != "default"
                    and agent.history[-1] is not getattr(self, "_history_before_actions", {}).get(tag)):
                obs, act, msg, params, info = agent.history[-1]
                info = dict(info or {})
                # The incoming receipt belongs to the previous decision.
                info.pop("action_receipts", None)
                info["execution_request"] = copy.deepcopy(request)
                agent.history[-1] = (obs, act, msg, params, info)

    def _publish_action_receipt(self, tag, request, **result):
        receipt = {**copy.deepcopy(request), **result}
        if tag in getattr(self, "infos", {}):
            receipts = self.infos[tag].setdefault("action_receipts", [])
            receipts[:] = [r for r in receipts if r.get("selection_id") != receipt.get("selection_id")]
            receipts.append(receipt)
        agent = getattr(self, "agents", {}).get(tag)
        if agent is not None:
            for index in range(len(getattr(agent, "history", [])) - 1, -1, -1):
                obs, act, msg, params, info = agent.history[index]
                if (info or {}).get("execution_request", {}).get("selection_id") == request.get("selection_id"):
                    info = {**info, "execution_receipt": copy.deepcopy(receipt)}
                    agent.history[index] = (obs, act, msg, params, info)
                    break
        path = getattr(self, "exp_logdir", None)
        if path is not None:
            with (path / "action_receipts.jsonl").open("a") as stream:
                stream.write(json.dumps({"agent": tag, **receipt}, default=str) + "\n")

    def _record_local_action_receipts(self, actions, external_state, ts):
        for tag, request in getattr(self, "_action_requests", {}).items():
            if tag in external_state.tools:
                continue
            info = self.infos.get(tag, {})
            result = {
                "response_step": ts,
                "executed_action": actions.get(tag, {}).get("action"),
                "response_status": "received",
                "effect_status": "unknown",
            }
            if "artifact_effect" in info:
                result.update(info["artifact_effect"])
            self._publish_action_receipt(tag, request, **result)

    def _record_external_outcomes(self, state, incoming_by_agent, ts):
        pending = getattr(self, "_pending_external_requests", None)
        if pending is None:
            pending = self._pending_external_requests = {}
        current_requests = set()

        for tag in sorted(state.dispatched):
            server, tool = state.tools[tag]
            status = (state.sent_statuses.get(tag) or [{}])[0]
            request_id = status.get("message_id") or ""
            topic, vote = state.vote_outcomes.get(tag, ("", None))
            participants = sorted(vote.contributors) if vote else [tag]
            selections = {
                peer: copy.deepcopy(getattr(self, "_action_requests", {}).get(peer, {
                    "selection_id": uuid.uuid4().hex, "step": ts,
                    "action": f"{server}_{tool}", "params": state.requests.get(peer, {}),
                })) for peer in set(participants) | {tag}
            }
            entry = {
                "step": ts, "agent": tag, "server": server, "tool": tool,
                "topic": "" if topic is None else str(topic),
                "params": state.requests.get(tag, {}),
                "coalition": ({"representative": vote.representative,
                               "contributors": sorted(vote.contributors),
                               "misaligned": sorted(vote.misaligned)} if vote else None),
                "message_id": request_id, "request_id": request_id,
                "delivery_status": status.get("status", ""),
            }
            envelope = {"outcome": entry, "selections": selections,
                        "call_context": getattr(state, "call_contexts", {}).get(tag, {}),
                        # The log's display topic is a string. Reward strategies
                        # must retain the original topic type across checkpoints.
                        "reward_topic": copy.deepcopy(topic if vote else tag),
                        "reward_winning_choice": vote.winning_choice if vote else "",
                        "reward_contract": reward_contract(getattr(self, "voting_managers", {}).get(server))}
            key = request_key(server, tag, request_id)
            if request_id:
                pending[key] = envelope
                current_requests.add(key)
            responses = incoming_by_agent.get(tag, {}).get(server, [])
            matched = [r for r in responses if isinstance(r, dict) and request_id
                       and r.get("message_id") == request_id
                       and r.get("response_complete", True)]
            if not matched:
                delivery = entry["delivery_status"]
                failed = not (delivery.startswith("sent") or delivery == "queued for delivery")
                self._finish_external_receipt(envelope, [], ts,
                                              response_status="delivery_failed" if failed else "pending")

        # Only an explicit request ID joins an arriving response to an action.
        # Uncorrelated pushes remain observations; they are never guessed here.
        # A correlated acknowledgement/progress event is not a completion: keep
        # its request pending so the later final reply retains reward ownership.
        for tag, servers in incoming_by_agent.items():
            for server, records in servers.items():
                grouped = {}
                for record in records:
                    if (isinstance(record, dict) and record.get("message_id")
                            and record.get("response_complete", True)):
                        grouped.setdefault(record["message_id"], []).append(record)
                for request_id, responses in grouped.items():
                    key = request_key(server, tag, request_id)
                    envelope = pending.get(key)
                    if envelope is None:
                        continue
                    failure = all(r.get("delivery_failed") for r in responses)
                    if key not in current_requests and not failure:
                        self._settle_deferred_reward(envelope, responses)
                    self._finish_external_receipt(envelope, responses, ts)
                    # A transport error leaves the world effect unresolved; a
                    # late, correlated reply may still resolve the same request.
                    if not failure:
                        pending.pop(key, None)

        aligned = set(state.dispatched)
        for _, vote in state.vote_outcomes.values():
            aligned |= vote.contributors
        for tag, params in state.requests.items():
            if tag in aligned:
                continue
            request = getattr(self, "_action_requests", {}).get(tag, {"step": ts, "params": params})
            self._publish_action_receipt(tag, request, response_step=ts,
                                         operation_status="overridden", effect_status="not_observed",
                                         note="This proposal was not sent to the world.")

    def _settle_deferred_reward(self, envelope, records):
        """Pay a correlated completion to its original vote, once.

        Current dispatches have already gone through manager.settle. A pending
        request has already paid its contribution costs, so only on_reward is
        repeated here. The pending envelope is checkpointed with the world.
        """
        if envelope.get("reward_settled"):
            return
        fields = envelope["outcome"]
        manager = getattr(self, "voting_managers", {}).get(fields["server"])
        contract = envelope.get("reward_contract")
        if (manager is None or contract is None or contract != reward_contract(manager)
                or "reward_topic" not in envelope):
            # Old checkpoints lack a typed topic/strategy contract. Do not
            # guess one from whichever election happens to be current now.
            envelope["reward_settlement_status"] = "original_reward_contract_unavailable"
            return
        coalition = fields.get("coalition")
        outcome = (VoteOutcome(
            winning_choice=envelope.get("reward_winning_choice", ""),
            representative=coalition["representative"],
            contributors=set(coalition["contributors"]),
            misaligned=set(coalition["misaligned"]),
        ) if coalition else VoteOutcome.solo(fields["agent"]))
        observations = {fields["agent"]: {"external_response": [r["text"] for r in records
            if r.get("response_complete", True) and r.get("text")
            and not r.get("tool_error") and not r.get("delivery_failed")]}}
        rewards = VotingManager._rewards_from(observations, fields["agent"])
        applied = envelope.get("rewards_applied", 0)
        for index, reward in enumerate(rewards):
            if index < applied:
                continue
            manager.rewards.on_reward(outcome, envelope["reward_topic"], reward, self.env)
            envelope["rewards_applied"] = index + 1
        envelope["reward_settled"] = True
        envelope["reward_settlement_status"] = "settled" if rewards else "no_usable_reward"

    def _finish_external_receipt(self, envelope, records, ts, response_status=None):
        successful = [r for r in records if not r.get("delivery_failed")]
        transport_warnings = [r.get("text", "") for r in records if r.get("delivery_failed")]
        if successful:
            records = successful
        texts = []
        for record in records:
            if record.get("structured_content") is not None:
                texts.append(record["structured_content"])
            if record.get("text"):
                text = record["text"]
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    parsed = text
                if parsed not in texts:
                    texts.append(parsed)
        if response_status is None:
            response_status = ("delivery_failed" if any(r.get("delivery_failed") for r in records)
                               else "tool_error" if any(r.get("tool_error") for r in records)
                               else "received" if texts else "empty")
        fields = envelope["outcome"]
        effect = {"effect_status": "unknown", "effect_evidence": {}}
        handler = getattr(self, "external_state_handlers", {}).get(fields["server"])
        if response_status == "received" and hasattr(handler, "assess_effect"):
            effect = handler.assess_effect(fields["tool"], fields["params"], envelope["call_context"], texts)
        for tag, request in envelope["selections"].items():
            self._publish_action_receipt(tag, request, request_id=fields["request_id"],
                response_step=ts, response_status=response_status, transport_warnings=transport_warnings, **effect,
                reward_settlement_status=envelope.get("reward_settlement_status"),
                executed_action=f"{fields['server']}_{fields['tool']}", executed_params=fields["params"])
