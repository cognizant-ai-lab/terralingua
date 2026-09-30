"""Execution evidence must survive delayed responses, voting and private history."""

import json
from types import SimpleNamespace

from terralingua.experiment.execution_receipts import ExecutionReceiptsMixin
from terralingua.experiment.runner import SimulationRunner, _ExternalState
from terralingua.voting.election import VoteOutcome
from terralingua.voting.manager import (
    IndependentVotingManager,
    UnanimousVotingManager,
    VotingManager,
)
from terralingua.voting.rewards import DiasRewards, DirectRewards


def runner(tmp_path):
    value = ExecutionReceiptsMixin()
    value.agents = {'a': SimpleNamespace(history=[({'energy': 1, 'time': 99}, 'world_act', 'keep social history', {}, {'custom': 'preserve'})])}
    value.infos = {'a': {}}
    value.external_state_handlers = {}
    return value


def call(identifier='req1', action='first'):
    return _ExternalState(requests={'a': {'action': action}}, tools={'a': ('world', 'act')},
                          dispatched={'a'}, sent_statuses={'a': [{'message_id': identifier, 'status': 'queued for delivery'}]})


def response(identifier, **fields):
    return {'message_id': identifier, 'text': json.dumps({'reward': 1, **fields})}


def test_late_response_cannot_be_assigned_to_new_action(tmp_path):
    value = runner(tmp_path)
    value._capture_action_requests({'a': {'action': 'world_act', 'params': {'action': 'first'}}}, 10)
    original = value._action_requests['a'].copy()
    value._record_external_outcomes(call(), {}, 10)
    assert value.infos['a']['action_receipts'][0]['response_status'] == 'pending'
    # Simulate checkpoint JSON round-trip before the second decision.
    value._pending_external_requests = json.loads(json.dumps(value._pending_external_requests))
    value._capture_action_requests({'a': {'action': 'world_act', 'params': {'action': 'second'}}}, 11)
    value._record_external_outcomes(call('req2', 'second'), {'a': {'world': [response('req1')]}}, 11)
    receipts = value.infos['a']['action_receipts']
    completed = [r for r in receipts if r['response_status'] == 'received']
    assert len(completed) == 1
    assert completed[0]['selection_id'] == original['selection_id']
    assert completed[0]['executed_params']['action'] == 'first'
    assert completed[0]['step'] == 10 and completed[0]['response_step'] == 11
    assert completed[0]['effect_status'] == 'unknown'
    assert [r['params']['action'] for r in receipts if r['response_status'] == 'pending'] == ['second']


def test_uncorrelated_push_does_not_certify_action(tmp_path):
    value = runner(tmp_path)
    value._record_external_outcomes(call(), {'a': {'world': [response('')]}}, 1)
    assert [r['response_status'] for r in value.infos['a']['action_receipts']] == ['pending']


def test_tool_error_is_unusable_even_with_reward_field(tmp_path):
    value = runner(tmp_path)
    value._record_external_outcomes(call(), {'a': {'world': [{**response('req1'), 'tool_error': True}]}}, 1)
    receipt = value.infos['a']['action_receipts'][0]
    assert receipt['response_status'] == 'tool_error' and receipt['effect_status'] == 'unknown'


def test_receipt_enriches_history_without_rewriting_memory_or_messages(tmp_path):
    value = runner(tmp_path)
    value.agents['a'].internal_memory = 'A belief I have not yet revised'
    value._capture_action_requests({'a': {'action': 'world_act', 'params': {}}}, 7)
    value._record_external_outcomes(call(), {'a': {'world': [response('req1')]}}, 7)
    history = value.agents['a'].history[0]
    assert history[2] == 'keep social history'
    assert history[4]['custom'] == 'preserve'
    assert history[4]['execution_receipt']['step'] == 7
    assert value.agents['a'].internal_memory == 'A belief I have not yet revised'


def test_old_history_cannot_receive_a_new_selection_receipt(tmp_path):
    value = runner(tmp_path)
    old = value.agents['a'].history[-1]
    value._history_before_actions = {'a': old}
    actions = {'a': {'action': 'wait', 'params': {}, 'source': 'llm'}}
    value._capture_action_requests(actions, 12)
    value._record_local_action_receipts(actions, _ExternalState(), 12)
    assert value.agents['a'].history[-1] is old
    assert 'execution_request' not in old[4] and 'execution_receipt' not in old[4]
    assert value.infos['a']['action_receipts'][0]['step'] == 12


def test_default_selection_never_relabels_private_history(tmp_path):
    value = runner(tmp_path)
    old = value.agents['a'].history[-1]
    value._history_before_actions = {'a': old}
    # Even a fresh parse-failure history row is not proof the model selected
    # the fallback that the host executed.
    fallback_history = ({'time': 12}, 'wait', 'unchanged story', {}, {'custom': 'fresh'})
    value.agents['a'].history.append(fallback_history)
    actions = {'a': {'action': 'wait', 'params': {}, 'source': 'default', 'default_reason': 'parse_failure'}}
    value._capture_action_requests(actions, 12)
    value._record_local_action_receipts(actions, _ExternalState(), 12)
    assert value.agents['a'].history == [old, fallback_history]
    assert all('execution_request' not in entry[4] for entry in value.agents['a'].history)
    assert value.infos['a']['action_receipts'][0]['step'] == 12


def test_mixed_transport_warning_does_not_hide_correlated_world_result(tmp_path):
    value = runner(tmp_path)
    value._capture_action_requests({'a': {'action': 'world_act', 'params': {'action': 'first'}}}, 10)
    failure = {'message_id': 'req1', 'text': 'Connection timed out before the reply arrived.', 'delivery_failed': True}
    incoming = {'a': {'world': [failure, response('req1')]}}
    value._record_external_outcomes(call(), incoming, 10)
    receipt = value.infos['a']['action_receipts'][-1]
    assert receipt['response_status'] == 'received'
    assert receipt['transport_warnings'] == [failure['text']]
    assert receipt['effect_status'] == 'unknown'
    assert not value._pending_external_requests
    # A duplicated callback must not produce a second result for this request.
    value._record_external_outcomes(_ExternalState(), {'a': {'world': [response('req1')]}}, 11)
    assert value.infos['a']['action_receipts'] == [receipt]


def test_tool_error_is_visible_without_poisoning_shared_state_or_reward(tmp_path):
    events = []
    handler = SimpleNamespace(observe=lambda values: events.extend(values))
    message = {'message_id': 'r1', 'text': '{"reward":99,"done":true}', 'tool_error': True}
    manager = SimpleNamespace(get_broadcast_messages=lambda: [], get_incoming_records=lambda tag: [message])
    value = SimpleNamespace(external_managers={'world': manager}, external_state_handlers={'world': handler},
                            obs={'a': {}}, infos={'a': {}},
                            env=SimpleNamespace(step_count=1, logger=SimpleNamespace(log=lambda **kwargs: None)))
    records = SimulationRunner._handle_incoming_external_messages(value, {})
    assert records['a']['world'][0] == message
    assert not events
    assert 'external_failure' in value.obs['a']['external_response'][0]
    assert VotingManager._rewards_from(value.obs, 'a') == []
    assert VotingManager._rewards_from({'a': {'external_response': ['{"reward":2,"error":"failed"}']}}, 'a') == []


def test_late_reward_uses_original_coalition_and_typed_topic_once_after_resume(tmp_path):
    value = runner(tmp_path)
    transfers = []
    value.env = SimpleNamespace(inject_resource=lambda **kwargs: transfers.append(kwargs))
    rewards = DiasRewards(cost=4, coefficient=0.2)
    rewards._reward_history[7].append(1.0)
    manager = UnanimousVotingManager(rewards)
    value.voting_managers = {'world': manager}
    original = VoteOutcome('original choice', 'a', {'a', 'b'}, {'c'})
    old_call = call()
    old_call.vote_outcomes = {'a': (7, original)}
    manager._outcomes = {7: original}
    # Normal current settlement already shares the contribution cost, even
    # though this request has not produced a reward yet.
    manager.settle({}, value.env)
    assert len(transfers) == 2
    value._record_external_outcomes(old_call, {}, 10)
    value._pending_external_requests = json.loads(json.dumps(value._pending_external_requests))
    # A later election is unrelated, even when the representative is the same.
    manager._outcomes = {'7': VoteOutcome('new choice', 'a', {'a', 'c'}, {'b'})}
    current = call('req2', 'second')
    current.vote_outcomes = {'a': ('7', manager._outcomes['7'])}
    value._record_external_outcomes(current, {'a': {'world': [response('req1', reward=3)]}}, 11)
    assert len(transfers) == 5  # original costs plus three original reward shares
    paid = {item['target']: item['amount'] for item in transfers[2:]}
    assert paid == {'a': 8.0, 'b': 8.0, 'c': -16.0}
    assert list(rewards._reward_history[7]) == [1.0, 3.0]
    assert '7' not in rewards._reward_history
    value._record_external_outcomes(_ExternalState(), {'a': {'world': [response('req1', reward=3)]}}, 12)
    assert len(transfers) == 5


def test_current_reward_is_not_paid_again_by_receipt_recorder(tmp_path):
    value = runner(tmp_path)
    transfers = []
    value.env = SimpleNamespace(inject_resource=lambda **kwargs: transfers.append(kwargs))
    manager = IndependentVotingManager(DirectRewards(2))
    value.voting_managers = {'world': manager}
    manager._callers = {'a': {}}
    manager.settle({'a': {'external_response': [response('req1')['text']]}}, value.env)
    value._record_external_outcomes(call(), {'a': {'world': [response('req1')]}}, 10)
    assert transfers == [{'amount': 1.0, 'coefficient': 2.0, 'target': 'a'}]


def test_deferred_reward_ignores_errors_but_recovers_after_transport_failure(tmp_path):
    value = runner(tmp_path)
    transfers = []
    value.env = SimpleNamespace(inject_resource=lambda **kwargs: transfers.append(kwargs))
    value.voting_managers = {'world': IndependentVotingManager(DirectRewards(2))}
    value._record_external_outcomes(call(), {}, 10)
    failure = {**response('req1', reward=99), 'delivery_failed': True}
    value._record_external_outcomes(_ExternalState(), {'a': {'world': [failure]}}, 11)
    assert not transfers and value._pending_external_requests
    value._record_external_outcomes(_ExternalState(), {'a': {'world': [failure, response('req1')]}}, 12)
    assert transfers == [{'amount': 1.0, 'coefficient': 2.0, 'target': 'a'}]
    assert not value._pending_external_requests
    for identifier, invalid in [('req2', {'tool_error': True}), ('req3', {'error': 'world action failed'})]:
        value._record_external_outcomes(call(identifier), {}, 13)
        record = (dict(response(identifier, reward=99), **invalid) if 'tool_error' in invalid
                  else response(identifier, reward=99, **invalid))
        value._record_external_outcomes(_ExternalState(), {'a': {'world': [record]}}, 14)
    assert len(transfers) == 1


def test_deferred_reward_does_not_guess_after_strategy_change(tmp_path):
    value = runner(tmp_path)
    transfers = []
    value.env = SimpleNamespace(inject_resource=lambda **kwargs: transfers.append(kwargs))
    value.voting_managers = {'world': IndependentVotingManager(DirectRewards(2))}
    value._record_external_outcomes(call(), {}, 10)
    value.voting_managers['world'] = IndependentVotingManager(DirectRewards(20))
    value._record_external_outcomes(_ExternalState(), {'a': {'world': [response('req1')]}}, 11)
    assert not transfers
    assert value.infos['a']['action_receipts'][-1]['reward_settlement_status'] == 'original_reward_contract_unavailable'


def test_reward_normalization_survives_json_checkpoint_with_typed_topics(tmp_path):
    original = runner(tmp_path)
    strategy = DiasRewards(4, 0.2)
    strategy._normalized_impact(7, 1.0)
    strategy._normalized_impact(7, 2.0)
    strategy._normalized_impact('7', 99.0)
    original.voting_managers = {'world': UnanimousVotingManager(strategy)}
    saved = json.loads(json.dumps(original._reward_state_checkpoint()))
    resumed = ExecutionReceiptsMixin()
    fresh = DiasRewards(4, 0.2)
    resumed.voting_managers = {'world': UnanimousVotingManager(fresh)}
    resumed._restore_reward_state(saved)
    assert fresh._normalized_impact(7, 3.0) == strategy._normalized_impact(7, 3.0)
    assert list(fresh._reward_history['7']) == [99.0]
    duplicate = '{"reward":5,"step":9}'
    assert VotingManager._rewards_from({'a': {'external_response': [duplicate, duplicate]}}, 'a') == [5.0]
