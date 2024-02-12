#r "nuget: dotenv.net, 3.1.2"

using dotenv.net;

// Read values from .env file
var envVars = DotEnv.Fluent()
    .WithoutExceptions()
    .WithEnvFiles("../../../.env")
    .WithTrimValues()
    .WithDefaultEncoding()
    .WithOverwriteExistingVars()
    .WithoutProbeForEnv()
    .Read();

// Load values into variables and strip quotes
var model = envVars["AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"].Replace("\"", "");
var azureEndpoint = envVars["AZURE_OPENAI_ENDPOINT"].Replace("\"", "");
var apiKey = envVars["AZURE_OPENAI_API_KEY"].Replace("\"", "");

#r "nuget: Microsoft.SemanticKernel, 1.0.1"

using Microsoft.SemanticKernel; 
var builder = Kernel.CreateBuilder(); 
builder.Services.AddAzureOpenAIChatCompletion(model, azureEndpoint, apiKey); 
var kernel = builder.Build();

var whatCanIMakeFunction = kernel.CreateFunctionFromPrompt( new PromptTemplateConfig() { 
    Template = @"What interesting things can I make with a {{$item}}?", 
    InputVariables = [ new() { 
        Name = "item", 
        Description = "An item to make something with.", 
        IsRequired=true } 
        ] 
    }
);

string item = "raspberry pi"; 
var response = await kernel.InvokeAsync(whatCanIMakeFunction, new () { { "item", item }}); 
Console.WriteLine(response);

var thingsToMakeSummary = kernel.CreateFunctionFromPrompt( new PromptTemplateConfig() { 
    Template = @"Summarize the following text: {{$thingsToMake}}?", 
    InputVariables = [ new() { 
        Name = "thingsToMake", 
        Description = "A list of things you could make.", 
        IsRequired=true } 
        ] 
    }
);

var summary = await kernel.InvokeAsync(thingsToMakeSummary, new () { { "thingsToMake", response }}); 
Console.WriteLine(summary);