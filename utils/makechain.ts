import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question that encompasses all necessary context from the chat history.

Chat History:
{chat_history}

Follow-up input: {question}

Make sure your standalone question is self-contained, clear, and specific. Rephrased standalone question:`;

// --------------------------------------------------

const QA_PROMPT = `You are an intelligent AI assistant designed to write novels based on specific provided documents. The context from these documents has been processed and made accessible to you. 

Your mission is to generate complete novels that are accurate, and comprehensive, drawing upon the information contained in the context of the documents. Those novels must be of 1500 words or more. If the answer isn't readily found in the documents, you should make use of your training data and understood context to infer and provide the most plausible response.

Here is the context from the documents:

Context: {context}

Here is the user's question:

Question: {question}

Provide your response in markdown format.`;

// Creates a ConversationalRetrievalQAChain object that uses an OpenAI model and a PineconeStore vectorstore
export const makeChain = (
  vectorstore: PineconeStore,
  returnSourceDocuments: boolean,
  modelTemperature: number,
  openAIapiKey: string,
) => {
  const model = new OpenAI({
    temperature: modelTemperature, // increase temepreature to get more creative answers
    // modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
    modelName: 'gpt-4', //change this to gpt-4 if you have access
    openAIApiKey: openAIapiKey,
  });

  // Configures the chain to use the QA_PROMPT and CONDENSE_PROMPT prompts and to not return the source documents
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      //questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments,
    },
  );
  return chain;
};
