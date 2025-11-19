const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');
const logger = require('../core/logger').createServiceLogger('LLM');
const config = require('../core/config');
const { promptLoader } = require('../../prompt-loader');

class LLMService {
  constructor() {
    this.client = null;
    this.geminiClient = null;
    this.isInitialized = false;
    this.requestCount = 0;
    this.errorCount = 0;
    this.useGemini = true; // Default to Gemini

    this.initializeClient();
  }

  initializeClient() {
    // Try to initialize Gemini first
    const geminiApiKey = process.env.GEMINI_API_KEY;

    if (geminiApiKey && geminiApiKey !== 'your-api-key-here') {
      try {
        this.geminiClient = new GoogleGenerativeAI(geminiApiKey);
        this.useGemini = true;
        this.isInitialized = true;

        logger.info('Gemini AI client initialized successfully', {
          model: 'gemini-1.5-flash'
        });
        return;
      } catch (error) {
        logger.error('Failed to initialize Gemini client', {
          error: error.message
        });
      }
    }

    // Fallback to OpenAI if Gemini not available
    const openaiApiKey = config.getApiKey('OPENAI');

    if (!openaiApiKey || openaiApiKey === 'your-api-key-here') {
      logger.warn('No API key configured (tried Gemini and OpenAI)', {
        geminiKeyExists: !!geminiApiKey,
        openaiKeyExists: !!openaiApiKey
      });
      return;
    }

    try {
      this.client = new OpenAI({
        apiKey: openaiApiKey,
        dangerouslyAllowBrowser: true
      });
      this.useGemini = false;
      this.isInitialized = true;

      logger.info('OpenAI client initialized successfully (fallback)', {
        model: config.get('llm.openai.model')
      });
    } catch (error) {
      logger.error('Failed to initialize OpenAI client', {
        error: error.message
      });
    }
  }

  getGenerationConfig(overrides = {}) {
    const defaults = config.get('llm.openai.generation') || {};
    const fallback = {
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 4096
    };

    const merged = { ...fallback, ...defaults, ...overrides };
    return Object.fromEntries(
      Object.entries(merged).filter(([, value]) => value !== undefined && value !== null)
    );
  }

  /**
   * Process an image directly with OpenAI using the active skill prompt.
   * @param {Buffer} imageBuffer - PNG/JPEG image bytes
   * @param {string} mimeType - e.g., 'image/png' or 'image/jpeg'
   * @param {string} activeSkill - current skill (e.g. 'dsa')
   * @param {Array} sessionMemory - optional (not required for image)
   * @param {string|null} programmingLanguage - optional language context for skills that need it
   * @returns {Promise<{response: string, metadata: object}>}
   */
  async processImageWithSkill(imageBuffer, mimeType, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check OpenAI API key configuration.');
    }

    if (!imageBuffer || !Buffer.isBuffer(imageBuffer)) {
      throw new Error('Invalid image buffer provided to processImageWithSkill');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      // Build system instruction using the skill prompt
      const { promptLoader } = require('../../prompt-loader');
      const skillPrompt = promptLoader.getSkillPrompt(activeSkill, programmingLanguage) || '';

      // Convert buffer to base64
      const base64 = imageBuffer.toString('base64');
      const imageUrl = `data:${mimeType};base64,${base64}`;

      const messages = [
        {
          role: 'system',
          content: skillPrompt || `You are an expert assistant for ${activeSkill.toUpperCase()} problems.`
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: this.formatImageInstruction(activeSkill, programmingLanguage)
            },
            {
              type: 'image_url',
              image_url: {
                url: imageUrl
              }
            }
          ]
        }
      ];

      const response = await this.executeRequest(messages);

      // Enforce language in code fences if provided
      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(response, programmingLanguage)
        : response;

      logger.logPerformance('LLM image processing', startTime, {
        activeSkill,
        imageSize: imageBuffer.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isImageAnalysis: true,
          mimeType
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM image processing failed', {
        error: error.message,
        activeSkill,
        requestId: this.requestCount
      });

      if (config.get('llm.openai.fallbackEnabled')) {
        return this.generateFallbackResponse('[image]', activeSkill);
      }
      throw error;
    }
  }

  formatImageInstruction(activeSkill, programmingLanguage) {
    const langNote = programmingLanguage ? ` Use only ${programmingLanguage.toUpperCase()} for any code.` : '';
    return `Analyze this image for a ${activeSkill.toUpperCase()} question. Extract the problem concisely and provide the best possible solution with explanation and final code.${langNote}`;
  }

  async processTextWithSkill(text, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check OpenAI API key configuration.');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      logger.info('Processing text with LLM', {
        activeSkill,
        textLength: text.length,
        hasSessionMemory: sessionMemory.length > 0,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      const messages = this.buildChatMessages(text, activeSkill, sessionMemory, programmingLanguage);
      const response = await this.executeRequest(messages);

      // Enforce language in code fences if programmingLanguage specified
      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(response, programmingLanguage)
        : response;

      logger.logPerformance('LLM text processing', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM processing failed', {
        error: error.message,
        activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      if (config.get('llm.openai.fallbackEnabled')) {
        return this.generateFallbackResponse(text, activeSkill);
      }

      throw error;
    }
  }

  async processTranscriptionWithIntelligentResponse(text, activeSkill, sessionMemory = [], programmingLanguage = null) {
    if (!this.isInitialized) {
      throw new Error('LLM service not initialized. Check OpenAI API key configuration.');
    }

    const startTime = Date.now();
    this.requestCount++;

    try {
      logger.info('Processing transcription with intelligent response', {
        activeSkill,
        textLength: text.length,
        hasSessionMemory: sessionMemory.length > 0,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      const messages = this.buildIntelligentTranscriptionMessages(text, activeSkill, sessionMemory, programmingLanguage);
      const response = await this.executeRequest(messages);

      // Enforce language in code fences if programmingLanguage specified
      const finalResponse = programmingLanguage
        ? this.enforceProgrammingLanguage(response, programmingLanguage)
        : response;

      logger.logPerformance('LLM transcription processing', startTime, {
        activeSkill,
        textLength: text.length,
        responseLength: finalResponse.length,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      return {
        response: finalResponse,
        metadata: {
          skill: activeSkill,
          programmingLanguage,
          processingTime: Date.now() - startTime,
          requestId: this.requestCount,
          usedFallback: false,
          isTranscriptionResponse: true
        }
      };
    } catch (error) {
      this.errorCount++;
      logger.error('LLM transcription processing failed', {
        error: error.message,
        activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        requestId: this.requestCount
      });

      if (config.get('llm.openai.fallbackEnabled')) {
        return this.generateIntelligentFallbackResponse(text, activeSkill);
      }

      throw error;
    }
  }

  /**
   * Normalize all triple-backtick code fences to the selected programming language tag.
   */
  enforceProgrammingLanguage(text, programmingLanguage) {
    try {
      if (!text || !programmingLanguage) return text;
      const norm = String(programmingLanguage).toLowerCase();
      const fenceTagMap = { cpp: 'cpp', c: 'c', python: 'python', java: 'java', javascript: 'javascript', js: 'javascript' };
      const fenceTag = fenceTagMap[norm] || norm || 'text';

      const replacedBackticks = text.replace(/```([^\n]*)\n/g, (match, info) => {
        const current = (info || '').trim();
        if (current.split(/\s+/)[0].toLowerCase() === fenceTag) return match;
        return '```' + fenceTag + '\n';
      });

      const normalizedTildes = replacedBackticks.replace(/~~~([^\n]*)\n/g, () => '```' + fenceTag + '\n');

      return normalizedTildes;
    } catch (_) {
      return text;
    }
  }

  buildChatMessages(text, activeSkill, sessionMemory, programmingLanguage) {
    const sessionManager = require('../managers/session.manager');

    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      const conversationHistory = sessionManager.getConversationHistory(15);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildChatMessagesWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage);
    }

    // Fallback to basic messages
    const requestComponents = promptLoader.getRequestComponents(
      activeSkill,
      text,
      sessionMemory,
      programmingLanguage
    );

    const messages = [];

    if (requestComponents.shouldUseModelMemory && requestComponents.skillPrompt) {
      messages.push({
        role: 'system',
        content: requestComponents.skillPrompt
      });

      logger.debug('Using language-enhanced system instruction for skill', {
        skill: activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        promptLength: requestComponents.skillPrompt.length
      });
    }

    messages.push({
      role: 'user',
      content: this.formatUserMessage(text, activeSkill)
    });

    return messages;
  }

  buildChatMessagesWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage) {
    const messages = [];

    if (skillContext.skillPrompt) {
      messages.push({
        role: 'system',
        content: skillContext.skillPrompt
      });

      logger.debug('Using skill context prompt as system instruction', {
        skill: activeSkill,
        programmingLanguage: programmingLanguage || 'not specified',
        promptLength: skillContext.skillPrompt.length
      });
    }

    // Add conversation history
    conversationHistory
      .filter(event => {
        return event.role !== 'system' &&
               event.content &&
               typeof event.content === 'string' &&
               event.content.trim().length > 0;
      })
      .forEach(event => {
        messages.push({
          role: event.role === 'model' ? 'assistant' : 'user',
          content: event.content.trim()
        });
      });

    // Add current user input
    const formattedMessage = this.formatUserMessage(text, activeSkill);
    if (!formattedMessage || formattedMessage.trim().length === 0) {
      throw new Error('Failed to format user message or message is empty');
    }

    messages.push({
      role: 'user',
      content: formattedMessage
    });

    logger.debug('Built chat messages with conversation history', {
      skill: activeSkill,
      programmingLanguage: programmingLanguage || 'not specified',
      historyLength: conversationHistory.length,
      totalMessages: messages.length
    });

    return messages;
  }

  buildIntelligentTranscriptionMessages(text, activeSkill, sessionMemory, programmingLanguage) {
    const cleanText = text && typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      throw new Error('Empty or invalid transcription text provided');
    }

    const sessionManager = require('../managers/session.manager');

    if (sessionManager && typeof sessionManager.getConversationHistory === 'function') {
      const conversationHistory = sessionManager.getConversationHistory(10);
      const skillContext = sessionManager.getSkillContext(activeSkill, programmingLanguage);
      return this.buildIntelligentTranscriptionMessagesWithHistory(cleanText, activeSkill, conversationHistory, skillContext, programmingLanguage);
    }

    // Fallback to basic intelligent request
    const messages = [];

    const intelligentPrompt = this.getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage);
    if (!intelligentPrompt) {
      throw new Error('Failed to generate intelligent transcription prompt');
    }

    messages.push({
      role: 'system',
      content: intelligentPrompt
    });

    messages.push({
      role: 'user',
      content: cleanText
    });

    logger.debug('Built basic intelligent transcription messages', {
      skill: activeSkill,
      programmingLanguage: programmingLanguage || 'not specified',
      textLength: cleanText.length
    });

    return messages;
  }

  buildIntelligentTranscriptionMessagesWithHistory(text, activeSkill, conversationHistory, skillContext, programmingLanguage) {
    const messages = [];

    // Use intelligent filter prompt
    const intelligentPrompt = this.getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage);
    messages.push({
      role: 'system',
      content: intelligentPrompt
    });

    // Add recent conversation history
    conversationHistory
      .filter(event => {
        return event.role !== 'system' &&
               event.content &&
               typeof event.content === 'string' &&
               event.content.trim().length > 0;
      })
      .slice(-8)
      .forEach(event => {
        const content = event.content.trim();
        if (content) {
          messages.push({
            role: event.role === 'model' ? 'assistant' : 'user',
            content: content
          });
        }
      });

    // Add current transcription
    const cleanText = text && typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      throw new Error('Empty or invalid transcription text provided');
    }

    messages.push({
      role: 'user',
      content: cleanText
    });

    if (messages.length === 0) {
      throw new Error('No valid content to send to OpenAI API');
    }

    logger.debug('Built intelligent transcription messages with conversation history', {
      skill: activeSkill,
      programmingLanguage: programmingLanguage || 'not specified',
      historyLength: conversationHistory.length,
      totalMessages: messages.length
    });

    return messages;
  }

  getIntelligentTranscriptionPrompt(activeSkill, programmingLanguage) {
    let prompt = `# Intelligent Transcription Response System

Assume you are asked a question in ${activeSkill.toUpperCase()} mode. Your job is to intelligently respond to question/message with appropriate brevity.
Assume you are in an interview and you need to perform best in ${activeSkill.toUpperCase()} mode.
Always respond to the point, do not repeat the question or unnecessary information which is not related to ${activeSkill}.`;

    if (programmingLanguage) {
      const lang = String(programmingLanguage).toLowerCase();
      const languageMap = { cpp: 'C++', c: 'C', python: 'Python', java: 'Java', javascript: 'JavaScript', js: 'JavaScript' };
      const fenceTagMap = { cpp: 'cpp', c: 'c', python: 'python', java: 'java', javascript: 'javascript', js: 'javascript' };
      const languageTitle = languageMap[lang] || (lang.charAt(0).toUpperCase() + lang.slice(1));
      const fenceTag = fenceTagMap[lang] || lang || 'text';
      prompt += `\n\nCODING CONTEXT: Respond ONLY in ${languageTitle}. All code blocks must use triple backticks with language tag \`\`\`${fenceTag}\`\`\`. Do not include other languages unless explicitly asked.`;
    }

    prompt += `

## Response Rules:

### If the transcription is casual conversation, greetings, or NOT related to ${activeSkill}:
- Respond with: "Yeah, I'm listening. Ask your question relevant to ${activeSkill}."
- Or similar brief acknowledgments like: "I'm here, what's your ${activeSkill} question?"

### If the transcription IS relevant to ${activeSkill} or is a follow-up question:
- Provide a comprehensive, detailed response
- Use bullet points, examples, and explanations
- Focus on actionable insights and complete answers
- Do not truncate or shorten your response

### Examples of casual/irrelevant messages:
- "Hello", "Hi there", "How are you?"
- "What's the weather like?"
- "I'm just testing this"
- Random conversations not related to ${activeSkill}

### Examples of relevant messages:
- Actual questions about ${activeSkill} concepts
- Follow-up questions to previous responses
- Requests for clarification on ${activeSkill} topics
- Problem-solving requests related to ${activeSkill}

## Response Format:
- Keep responses detailed
- Use bullet points for structured answers
- Be encouraging and helpful
- Stay focused on ${activeSkill}

If the user's input is a coding or DSA problem statement and contains no code, produce a complete, runnable solution in the selected programming language without asking for more details. Always include the final implementation in a properly tagged code block.

Remember: Be intelligent about filtering - only provide detailed responses when the user actually needs help with ${activeSkill}.`;

    return prompt;
  }

  formatUserMessage(text, activeSkill) {
    return `Context: ${activeSkill.toUpperCase()} analysis request\n\nText to analyze:\n${text}`;
  }

  async executeRequest(messages) {
    if (this.useGemini && this.geminiClient) {
      return await this.executeGeminiRequest(messages);
    } else {
      return await this.executeOpenAIRequest(messages);
    }
  }

  async executeGeminiRequest(messages) {
    const maxRetries = 3;
    const timeout = 30000;

    logger.debug('Executing Gemini request', {
      messageCount: messages.length,
      model: 'gemini-1.5-flash',
      timeout,
      maxRetries
    });

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const model = this.geminiClient.getGenerativeModel({ model: 'gemini-1.5-flash' });

        // Convert OpenAI message format to Gemini format
        const systemMessage = messages.find(m => m.role === 'system');
        const userMessages = messages.filter(m => m.role === 'user' || m.role === 'assistant');

        // Build conversation history for Gemini
        const history = [];
        for (let i = 0; i < userMessages.length - 1; i++) {
          const msg = userMessages[i];
          history.push({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content) }]
          });
        }

        // Start chat with history
        const chat = model.startChat({
          history: history,
          generationConfig: {
            temperature: 0.7,
            topP: 0.9,
            maxOutputTokens: 4096,
          },
          systemInstruction: systemMessage ? systemMessage.content : undefined
        });

        // Send the latest message
        const lastMessage = userMessages[userMessages.length - 1];
        const lastContent = typeof lastMessage.content === 'string'
          ? lastMessage.content
          : JSON.stringify(lastMessage.content);

        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Request timeout')), timeout)
        );

        const requestPromise = chat.sendMessage(lastContent);
        const result = await Promise.race([requestPromise, timeoutPromise]);

        const responseText = result.response.text();

        logger.debug('Gemini API request successful', {
          attempt,
          responseLength: responseText.length
        });

        return responseText;
      } catch (error) {
        logger.warn(`Gemini API attempt ${attempt} failed`, {
          error: error.message,
          remainingAttempts: maxRetries - attempt
        });

        if (attempt === maxRetries) {
          throw new Error(`Gemini API failed after ${maxRetries} attempts: ${error.message}`);
        }

        const delay = 1500 * attempt + Math.random() * 1000;
        await this.delay(delay);
      }
    }
  }

  async executeOpenAIRequest(messages) {
    const maxRetries = config.get('llm.openai.maxRetries');
    const timeout = config.get('llm.openai.timeout');
    const model = config.get('llm.openai.model');
    const generationConfig = this.getGenerationConfig();

    logger.debug('Executing OpenAI request', {
      hasClient: !!this.client,
      messageCount: messages.length,
      model,
      timeout,
      maxRetries
    });

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Request timeout')), timeout)
        );

        logger.debug(`OpenAI API attempt ${attempt} starting`, {
          timestamp: new Date().toISOString(),
          timeout
        });

        const requestPromise = this.client.chat.completions.create({
          model: model,
          messages: messages,
          ...generationConfig
        });

        const result = await Promise.race([requestPromise, timeoutPromise]);

        if (!result.choices || result.choices.length === 0) {
          throw new Error('Empty response from OpenAI API');
        }

        const responseText = result.choices[0].message.content;

        logger.debug('OpenAI API request successful', {
          attempt,
          responseLength: responseText.length,
          finishReason: result.choices[0].finish_reason
        });

        return responseText;
      } catch (error) {
        const errorInfo = this.analyzeError(error);

        logger.warn(`OpenAI API attempt ${attempt} failed`, {
          error: error.message,
          errorType: errorInfo.type,
          isNetworkError: errorInfo.isNetworkError,
          suggestedAction: errorInfo.suggestedAction,
          remainingAttempts: maxRetries - attempt
        });

        if (attempt === maxRetries) {
          const finalError = new Error(`OpenAI API failed after ${maxRetries} attempts: ${error.message}`);
          finalError.errorAnalysis = errorInfo;
          finalError.originalError = error;
          throw finalError;
        }

        const baseDelay = errorInfo.isNetworkError ? 2500 : 1500;
        const delay = baseDelay * attempt + Math.random() * 1000;

        logger.debug(`Waiting ${delay}ms before retry ${attempt + 1}`, {
          baseDelay,
          isNetworkError: errorInfo.isNetworkError
        });

        await this.delay(delay);
      }
    }
  }

  analyzeError(error) {
    const errorMessage = error.message.toLowerCase();

    if (errorMessage.includes('fetch failed') ||
        errorMessage.includes('network error') ||
        errorMessage.includes('enotfound') ||
        errorMessage.includes('econnrefused') ||
        errorMessage.includes('timeout')) {
      return {
        type: 'NETWORK_ERROR',
        isNetworkError: true,
        suggestedAction: 'Check internet connection and firewall settings'
      };
    }

    if (errorMessage.includes('unauthorized') ||
        errorMessage.includes('invalid api key') ||
        errorMessage.includes('incorrect api key') ||
        errorMessage.includes('forbidden')) {
      return {
        type: 'AUTH_ERROR',
        isNetworkError: false,
        suggestedAction: 'Verify OpenAI API key configuration'
      };
    }

    if (errorMessage.includes('quota') ||
        errorMessage.includes('rate limit') ||
        errorMessage.includes('too many requests')) {
      return {
        type: 'RATE_LIMIT_ERROR',
        isNetworkError: false,
        suggestedAction: 'Wait before retrying or check API quota'
      };
    }

    if (errorMessage.includes('request timeout') || errorMessage.includes('etimedout')) {
      return {
        type: 'TIMEOUT_ERROR',
        isNetworkError: true,
        suggestedAction: 'Check network latency or increase timeout'
      };
    }

    return {
      type: 'UNKNOWN_ERROR',
      isNetworkError: false,
      suggestedAction: 'Check logs for more details'
    };
  }

  generateFallbackResponse(text, activeSkill) {
    logger.info('Generating fallback response', { activeSkill });

    const fallbackResponses = {
      'dsa': 'This appears to be a data structures and algorithms problem. Consider breaking it down into smaller components and identifying the appropriate algorithm or data structure to use.',
      'system-design': 'For this system design question, consider scalability, reliability, and the trade-offs between different architectural approaches.',
      'programming': 'This looks like a programming challenge. Focus on understanding the requirements, edge cases, and optimal time/space complexity.',
      'default': 'I can help analyze this content. Please ensure your OpenAI API key is properly configured for detailed analysis.'
    };

    const response = fallbackResponses[activeSkill] || fallbackResponses.default;

    return {
      response,
      metadata: {
        skill: activeSkill,
        processingTime: 0,
        requestId: this.requestCount,
        usedFallback: true
      }
    };
  }

  generateIntelligentFallbackResponse(text, activeSkill) {
    logger.info('Generating intelligent fallback response for transcription', { activeSkill });

    const skillKeywords = {
      'dsa': ['algorithm', 'data structure', 'array', 'tree', 'graph', 'sort', 'search', 'complexity', 'big o'],
      'programming': ['code', 'function', 'variable', 'class', 'method', 'bug', 'debug', 'syntax'],
      'system-design': ['scalability', 'database', 'architecture', 'microservice', 'load balancer', 'cache'],
      'behavioral': ['interview', 'experience', 'situation', 'leadership', 'conflict', 'team'],
      'sales': ['customer', 'deal', 'negotiation', 'price', 'revenue', 'prospect'],
      'presentation': ['slide', 'audience', 'public speaking', 'presentation', 'nervous'],
      'data-science': ['data', 'model', 'machine learning', 'statistics', 'analytics', 'python', 'pandas'],
      'devops': ['deployment', 'ci/cd', 'docker', 'kubernetes', 'infrastructure', 'monitoring'],
      'negotiation': ['negotiate', 'compromise', 'agreement', 'terms', 'conflict resolution']
    };

    const textLower = text.toLowerCase();
    const relevantKeywords = skillKeywords[activeSkill] || [];
    const hasRelevantKeywords = relevantKeywords.some(keyword => textLower.includes(keyword));

    const questionIndicators = ['how', 'what', 'why', 'when', 'where', 'can you', 'could you', 'should i', '?'];
    const seemsLikeQuestion = questionIndicators.some(indicator => textLower.includes(indicator));

    let response;
    if (hasRelevantKeywords || seemsLikeQuestion) {
      response = `I'm having trouble processing that right now, but it sounds like a ${activeSkill} question. Could you rephrase or ask more specifically about what you need help with?`;
    } else {
      response = `Yeah, I'm listening. Ask your question relevant to ${activeSkill}.`;
    }

    return {
      response,
      metadata: {
        skill: activeSkill,
        processingTime: 0,
        requestId: this.requestCount,
        usedFallback: true,
        isTranscriptionResponse: true
      }
    };
  }

  async testConnection() {
    if (!this.isInitialized) {
      return { success: false, error: 'Service not initialized' };
    }

    try {
      const testMessages = [{
        role: 'user',
        content: 'Test connection. Please respond with "OK".'
      }];

      const startTime = Date.now();
      const result = await this.client.chat.completions.create({
        model: config.get('llm.openai.model'),
        messages: testMessages,
        max_tokens: 10,
        temperature: 0
      });

      const latency = Date.now() - startTime;
      const responseText = result.choices[0].message.content;

      logger.info('Connection test successful', {
        response: responseText,
        latency
      });

      return {
        success: true,
        response: responseText,
        latency
      };
    } catch (error) {
      const errorAnalysis = this.analyzeError(error);
      logger.error('Connection test failed', {
        error: error.message,
        errorAnalysis
      });

      return {
        success: false,
        error: error.message,
        errorAnalysis
      };
    }
  }

  updateApiKey(newApiKey) {
    process.env.OPENAI_API_KEY = newApiKey;
    this.isInitialized = false;
    this.initializeClient();

    logger.info('API key updated and client reinitialized');
  }

  getStats() {
    return {
      isInitialized: this.isInitialized,
      requestCount: this.requestCount,
      errorCount: this.errorCount,
      successRate: this.requestCount > 0 ? ((this.requestCount - this.errorCount) / this.requestCount) * 100 : 0,
      config: config.get('llm.openai')
    };
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = new LLMService();
