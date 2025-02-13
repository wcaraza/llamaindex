#context = """Purpose: The primary role of this agent is to bring recomendations by analyzing text data and other sources. It should be able to generate analisys from historical data."""

context = """
You are an Academic Advisor at {university_name}, guiding students on course selection, academic policies, and university resources. Keep responses short, friendly, and clear, using bullet points when helpful. Offer course recommendations based on students’ majors and interests, explain registration, graduation, and grading policies, and suggest academic support services like tutoring or career counseling. Help students manage their workload, stay motivated, and find scholarships or extracurricular opportunities. If a question requires official details, direct students to the appropriate university resources. Your goal is to make their academic journey easier and more enjoyable! 
"""
#context = """
#You are an expert academic analyst. Analyze the following student data and offer personalized recommendations.
#Consider factors such as performance, attendance, and participation.
#Respond with clear actions to improve student academic performance.
#Previous taken actions are placed on name column.
#"""
#analice las columnas y encuentre los alumnos mas parecidos y de acciones recomendadas para los estudiantes

#Purpose: The primary role of this agent is to assist users by analyzing code. It should be able to generate code and answer questions about code provided.

#code_parser_template = """Parse the response from a previous LLM into a description and a string of valid code,
#                            also come up with a valid filename this could be saved as that doesnt contain special characters.
#                            Here is the response: {response}. You should parse this in the following JSON Format: """

code_parser_template = """
Parse the response from a previous LLM into a description, here is the response: {response}. You should parse this in the following JSON Format: 
"""

student_ids = [296351, 252348]
#, 252348, 262380, 280038, 273860, 293026, 220207, 264045, 304893, 306941, 275510, 292036, 294883, 241586, 294121, 274720, 302787, 240332, 292446, 305689, 233838, 259125, 248750, 311064, 261782, 294570, 256899, 243270, 313731, 308912, 318325, 310727, 294153, 262025, 279178, 307456, 277534, 293683, 300398, 55647, 308912, 266416, 187465, 293123, 294839, 268685, 234386, 263711, 302576, 307456, 307011, 271470, 278725, 262996, 309698, 122011, 277187, 304055, 296812, 293741, 277665, 301212, 267854, 292463, 275688, 310974, 293169, 310423, 270208, 293807, 276466, 296463, 308425, 307203, 279305, 264045, 304598, 296184, 302472, 121665, 307862, 268791, 269595, 293840, 244521, 295076, 276126, 252342, 271955, 272742, 312868, 302255, 268594, 302488, 252010, 313379, 243943, 313787, 299847, 307715]
