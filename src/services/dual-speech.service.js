// Dual Audio Speech Service - Captures both microphone and system audio
// Uses VoiceMeeter to separate user voice from other person's voice

const sdk = require('microsoft-cognitiveservices-speech-sdk');
const { EventEmitter } = require('events');
const logger = require('../core/logger').createServiceLogger('DUAL-SPEECH');
const config = require('../core/config');

class DualSpeechService extends EventEmitter {
  constructor() {
    super();

    // Two separate recognizers - one for user, one for other person
    this.userRecognizer = null;
    this.otherRecognizer = null;

    this.isRecording = false;
    this.speechConfig = null;
    this.available = false;

    this.initializeClient();
  }

  initializeClient() {
    try {
      const subscriptionKey = process.env.AZURE_SPEECH_KEY;
      const region = process.env.AZURE_SPEECH_REGION;

      if (!subscriptionKey || !region) {
        logger.warn('Azure Speech credentials not found. Dual speech disabled.');
        this.available = false;
        this.emit('status', 'Dual speech recognition disabled (missing credentials)');
        return;
      }

      // Initialize Azure Speech configuration
      this.speechConfig = sdk.SpeechConfig.fromSubscription(subscriptionKey, region);
      this.speechConfig.speechRecognitionLanguage = 'en-US';
      this.speechConfig.outputFormat = sdk.OutputFormat.Detailed;

      // Optimize for continuous conversation
      this.speechConfig.setProperty(sdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000");
      this.speechConfig.setProperty(sdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500");
      this.speechConfig.setProperty(sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1500");

      logger.info('Dual Audio Speech service initialized', { region });
      this.available = true;
      this.emit('status', 'Dual audio capture ready');

    } catch (error) {
      logger.error('Failed to initialize dual speech service', { error: error.message });
      this.available = false;
      this.emit('status', 'Dual speech unavailable');
    }
  }

  async startDualRecording() {
    try {
      if (!this.speechConfig) {
        throw new Error('Speech service not initialized');
      }

      if (this.isRecording) {
        logger.warn('Dual recording already in progress');
        return;
      }

      logger.info('Starting dual audio capture...');
      this.isRecording = true;
      this.emit('recording-started');

      // Start the recognizer (VoiceMeeter provides mixed audio)
      await this._startUserRecognizer();

    } catch (error) {
      logger.error('Failed to start dual recording', { error: error.message });
      this.emit('error', `Dual recording failed: ${error.message}`);
      this.isRecording = false;
    }
  }

  async _startUserRecognizer() {
    try {
      // Use VoiceMeeter Out A1 to capture both microphone and system audio
      // VoiceMeeter mixes both streams into one output
      const audioConfig = sdk.AudioConfig.fromDefaultMicrophoneInput();
      this.userRecognizer = new sdk.SpeechRecognizer(this.speechConfig, audioConfig);

      logger.info('[USER] Using VoiceMeeter audio device for mixed microphone and system audio');

      // Set up event handlers for USER audio
      this.userRecognizer.recognizing = (s, e) => {
        logger.info('[USER] Recognizing event fired', {
          reason: e.result.reason,
          text: e.result.text
        });
        if (e.result.reason === sdk.ResultReason.RecognizingSpeech) {
          logger.info('[MIXED] Interim transcription', { text: e.result.text });
          this.emit('interim-transcription', {
            source: 'mixed',
            text: e.result.text,
            isFinal: false,
            speaker: 'Audio'
          });
        }
      };

      this.userRecognizer.recognized = (s, e) => {
        if (e.result.reason === sdk.ResultReason.RecognizedSpeech && e.result.text.trim()) {
          logger.info('[MIXED] Final transcription', { text: e.result.text });
          this.emit('transcription', {
            source: 'mixed',
            text: e.result.text,
            isFinal: true,
            speaker: 'Audio'
          });
        }
      };

      this.userRecognizer.canceled = (s, e) => {
        logger.warn('[USER] Recognition canceled', {
          reason: e.reason,
          errorCode: e.errorCode,
          errorDetails: e.errorDetails
        });
        if (e.reason === sdk.CancellationReason.Error) {
          logger.error('[USER] Recognition error', { details: e.errorDetails });
          this.emit('error', `User microphone error: ${e.errorDetails}`);
        }
      };

      this.userRecognizer.sessionStarted = (s, e) => {
        logger.info('[USER] Recognition session started - Audio stream connected');
      };

      this.userRecognizer.sessionStopped = (s, e) => {
        logger.info('[USER] Recognition session stopped');
      };

      this.userRecognizer.speechStartDetected = (s, e) => {
        logger.info('[USER] Speech start detected - Microphone is picking up voice');
      };

      this.userRecognizer.speechEndDetected = (s, e) => {
        logger.info('[USER] Speech end detected');
      };

      // Start continuous recognition for user
      await new Promise((resolve, reject) => {
        this.userRecognizer.startContinuousRecognitionAsync(
          () => {
            logger.info('[USER] Continuous recognition started');
            resolve();
          },
          (error) => {
            logger.error('[USER] Failed to start recognition', { error: error.toString() });
            reject(new Error(`User recognition failed: ${error}`));
          }
        );
      });

    } catch (error) {
      logger.error('Failed to start user recognizer', { error: error.message });
      throw error;
    }
  }

  async _startOtherRecognizer() {
    try {
      // For system audio capture (other person's voice from calls)
      // Using VoiceMeeter Output or Windows WASAPI Loopback

      // Option 1: Use default speakers (will capture all system audio)
      // NOTE: This requires special configuration - Windows doesn't allow easy capture of speaker output
      // We'll need to configure VoiceMeeter to route system audio to a virtual input

      logger.warn('[OTHER] System audio capture requires VoiceMeeter configuration');
      logger.warn('[OTHER] Please configure VoiceMeeter to route system audio to a virtual cable');

      // For now, we'll use a second microphone input or skip
      // TODO: Implement WASAPI Loopback or VoiceMeeter virtual cable

      this.emit('status', 'System audio capture requires manual VoiceMeeter setup');

      // Placeholder - will be implemented after VoiceMeeter configuration

    } catch (error) {
      logger.error('Failed to start other recognizer', { error: error.message });
      // Don't throw - user recognition can still work
    }
  }

  async stopDualRecording() {
    if (!this.isRecording) {
      return;
    }

    logger.info('Stopping dual audio capture...');
    this.isRecording = false;

    // Stop both recognizers
    await Promise.all([
      this._stopRecognizer(this.userRecognizer, 'USER'),
      this._stopRecognizer(this.otherRecognizer, 'OTHER')
    ]);

    this.emit('recording-stopped');
    this.emit('status', 'Dual recording stopped');
  }

  async _stopRecognizer(recognizer, label) {
    if (!recognizer) return;

    return new Promise((resolve) => {
      try {
        recognizer.stopContinuousRecognitionAsync(
          () => {
            logger.info(`[${label}] Recognition stopped successfully`);
            recognizer.close();
            resolve();
          },
          (error) => {
            logger.error(`[${label}] Error stopping recognition`, { error: error.toString() });
            try {
              recognizer.close();
            } catch (e) {
              // Ignore close errors
            }
            resolve();
          }
        );
      } catch (error) {
        logger.error(`[${label}] Error in stop process`, { error: error.message });
        resolve();
      }
    });
  }

  isAvailable() {
    return this.available && !!this.speechConfig;
  }

  getStatus() {
    return {
      isRecording: this.isRecording,
      isInitialized: !!this.speechConfig,
      available: this.available,
      userRecognizer: !!this.userRecognizer,
      otherRecognizer: !!this.otherRecognizer
    };
  }
}

module.exports = new DualSpeechService();
