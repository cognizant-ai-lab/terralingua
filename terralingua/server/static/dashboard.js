import { deselectAgent } from './js/roster.js';
import { initArtifacts } from './js/artifacts.js';
import { initDrawers } from './js/drawers.js';
import { wsRequest, connectDashboardWS } from './js/ws.js';
import { initAnthropologist } from './js/anthropologist.js';
import { checkSession, initSettingsDrawer } from './js/auth.js';
import { initTutorial } from './js/tutorial.js';
import { initGraphReplay } from './js/graph_replay.js';
import './js/controls.js';

initGraphReplay(wsRequest);
initArtifacts(wsRequest);
initDrawers(wsRequest, deselectAgent);
initAnthropologist(wsRequest);
await checkSession();
initSettingsDrawer();
initTutorial();
connectDashboardWS();
