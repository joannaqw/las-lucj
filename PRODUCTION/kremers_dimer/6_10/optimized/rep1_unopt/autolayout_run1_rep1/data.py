from qiskit_ibm_runtime import QiskitRuntimeService
job_id='d828scvoha1c73bl9beg'
service =  QiskitRuntimeService()
job = service.job(job_id)
job_result = job.result()
pub_result = job_result[0]
pub_result = job_result[0]

print(pub_result)
print(pub_result.data)
print(pub_result.data.keys() if hasattr(pub_result.data, "keys") else "no keys")
print(dir(pub_result.data))
